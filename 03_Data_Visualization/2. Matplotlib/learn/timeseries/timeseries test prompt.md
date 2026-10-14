Generate 100 New Objective Questions from the Reference Time Series Material

Use the attached/reference Jupyter Notebook and Markdown file as the primary source of truth for the topic, concepts, terminology, difficulty level, and question style.

Reference files:

- Reference Exam: [test.md](file:///d:/Data/GitHub/AI-ML-Study-Notes/basics/3.%20Mathplotlib/test.md)

Task

Analyze the reference notebooks and test file carefully and generate 100 NEW multiple-choice objective questions focused specifically on Matplotlib Temporal & Time Series Visualization (`matplotlib.dates` Locators & Formatters, `pd.date_range` Integration, `autofmt_xdate` Overlap Prevention, Dual-Axis Time Series via `twinx()`, Rolling Window Overlays, Confidence Interval Shading `fill_between`, Event Shading via `axvspan`, Trading Day Weekend Gaps, and ML Forecast Diagnostic Visualizations).

The 100 questions must be divided into 10 separate Markdown files, with exactly 10 questions per file.

Create in `basics/3. Mathplotlib/timeseries/`:

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

1. Use the Reference Material Correctly

First, thoroughly analyze the supplied `12. dates.ipynb`, `1. basic.ipynb`, and `test.md`.

Identify:

- Topics and subtopics covered:
  - Temporal Data Representations:
    - Pandas `pd.date_range("YYYY-MM-DD", periods=..., freq=...)`, `pd.DatetimeIndex`, `pd.to_datetime()`
    - Standard library `datetime.date`, `datetime.datetime`
    - NumPy `np.datetime64` arrays
    - Matplotlib internal date representation: float days since `0001-01-01 00:00:00 UTC` via `mdates.date2num()` and `mdates.num2date()`
  - Date Locators (`matplotlib.dates` / `mdates`):
    - Major Locators: `mdates.YearLocator()`, `mdates.MonthLocator(interval=1)`, `mdates.WeekdayLocator(byweekday=mdates.MO)`
    - Minor Locators: `mdates.DayLocator(interval=7)`, `mdates.HourLocator()`, `mdates.MinuteLocator()`
    - Attaching locators: `ax.xaxis.set_major_locator(...)`, `ax.xaxis.set_minor_locator(...)`
  - Date Formatters (`mdates.DateFormatter`):
    - Standard format strings: `%Y` (4-digit year), `%y` (2-digit year), `%m` (month 01–12), `%b` (short month, e.g. "Jan"), `%B` (full month, e.g. "January"), `%d` (day 01–31), `%H:%M:%S`
    - Multi-unit formatting: `mdates.DateFormatter("%b %Y")`
    - Modern dynamic formatter: `mdates.ConciseDateFormatter(locator)`
  - Overlap Prevention & Tick Alignment:
    - Automated date formatting: `fig.autofmt_xdate(rotation=45, ha='right')`
    - Manual rotation and alignment via `ax.tick_params(axis='x', rotation=45)`
    - Why `ha='right'` prevents misaligned label anchors
  - Temporal Axis Limits & Windowing:
    - Setting explicit timestamp limits: `ax.set_xlim(pd.to_datetime("2026-01-01"), pd.to_datetime("2026-06-30"))`
    - Passing string vs. Timestamp objects
  - Dual-Axis Temporal Visualizations (`ax.twinx()`):
    - Overlaying two time series with divergent units (e.g., Stock Price [$] vs. Trading Volume [units])
    - Sharing the identical x-axis date ruler
    - Combining handles and labels for a unified legend across both twin axes
  - Trend Lines, Smoothing, and Rolling Windows:
    - Overlaying raw noisy measurements with rolling statistics (`df['val'].rolling(window=7).mean()`)
    - Exponential smoothing curves
  - Uncertainty Bounds & Forecasting Envelopes:
    - Prediction interval shading: `ax.fill_between(dates, lower_bound, upper_bound, alpha=0.25, label="95% Forecast Interval")`
    - Actual vs. Predicted curves
  - Event Highlighting and Regime Shifts:
    - Vertical time span shading: `ax.axvspan(start_date, end_date, color='red', alpha=0.2, label='Anomaly Period')`
    - Vertical milestone markers: `ax.axvline(pd.to_datetime("2026-03-15"), color='black', linestyle='--')`
    - Timestamp-anchored annotations: `ax.annotate('System Outage', xy=(incident_date, value), xytext=(...), arrowprops=...)`
  - Discrete Financial Data & Weekend Gaps:
    - Non-continuous calendar axes: why plotting financial stock prices against a continuous date axis shows artificial flat-line gaps during weekends and holidays
    - Mitigation: plotting against integer indices with custom date string tick labels
  - Machine Learning & Forecasting Use Cases:
    - Chronological train / validation / test splits visualization (color-coded contiguous blocks)
    - Model residual error time series (evaluating autocorrelation and heteroscedasticity over time)
    - Multi-step ahead forecasting comparisons (ARIMA, Prophet, LSTM, LightGBM)
- Important definitions:
  - mdates, MonthLocator, DayLocator, DateFormatter, ConciseDateFormatter, autofmt_xdate, twinx, date2num, num2date, axvspan, fill_between
- Common mistakes and misconceptions:
  - Passing raw date strings (e.g., `["2026-01-01", "2026-01-02"]`) to `ax.plot()` expecting Matplotlib to treat them as dates rather than categorical strings, causing tick locator failures
  - Forgetting `fig.autofmt_xdate()`, resulting in completely illegible, overlapping horizontal date labels
  - Creating twin axes with `ax.twinx()` and calling `ax.legend()` on each, producing two overlapping legends inside the plot
  - Using `ax.set_xlim("2026-01-01", "2026-06-01")` without converting strings to `pd.to_datetime()` or `datetime` objects on numeric date axes
  - Confusing `%m` (month as number) with `%M` (minute) in `DateFormatter`
  - Assuming continuous date axes automatically filter out market closure weekends
- Question structure, option style, and explanation depth
- Correct-answer format and interactive presentation

Use these as the foundation for generating new questions.

Do not simply copy, paraphrase, or slightly modify the existing questions.

The generated questions must test the same underlying knowledge while introducing new datasets, model metrics, numerical values, code patterns, and reasoning requirements.

2. Generate Completely New Questions

All 100 questions must be meaningfully different from the questions in the reference files.

Avoid:

- Rephrasing an existing question without changing what it tests
- Simply changing the date range from 90 days to 180 days without changing the analytical demand
- Asking the same `%b` date format string question repeatedly
- Creating trivial true/false style questions disguised as multiple choice

Instead, create questions that require the learner to apply, calculate, interpret, debug, or reason about temporal locators, formatters, twin axes, and forecasting visualizations.

3. Maintain Real Rigor and Balance

Ensure the questions reflect authentic scenarios in data science, quantitative finance, and machine learning forecasting:

- Reading and predicting the visual output of realistic time series code snippets
- Calculating date interval ticks, formatting directives, and twin-axis coordinates
- Identifying date string type bugs, weekend gap artifacts, and overlapping date tick labels
- Selecting the correct visualization tools for forecast intervals, rolling averages, and anomaly shading

Difficulty Breakdown across the 100 questions:

- Conceptual / Architectural Understanding: ~30 questions
- Code Reading, Output Prediction & Date Math: ~40 questions
- Diagnostic Scenarios, Bug Fixing & Production Forecasting: ~30 questions

4. Answer Uniformity and Quality

To eliminate test-taking bias and guessing artifacts:

- Randomize and strictly balance the correct answer options across the 100 questions:
  - Approximately 25 questions with Option A as correct
  - Approximately 25 questions with Option B as correct
  - Approximately 25 questions with Option C as correct
  - Approximately 25 questions with Option D as correct
- Ensure distractor options are realistic, plausible, and represent common programmer errors
- Keep the length, technical depth, and grammatical structure of all four choices balanced so the correct choice never stands out by length alone

5. Question Style and Content Guidelines

- Every question must feature four clear choices: A, B, C, and D
- Code blocks must follow modern Python 3, Pandas, and Matplotlib conventions
- All calculations (e.g., date intervals, format tokens, timestamp bounds) must be mathematically precise
- Code snippets should model real ML workflow tasks: training loss curves, ROC-AUC comparisons, multi-axes hyperparameter heatmaps, and publication graphics

6. Prohibited Question Types (Anti-Paraphrasing Rules)

Do not include:

- Direct rephrasing of questions from `test.md`
- Vague trivia questions (e.g., "In what year was ConciseDateFormatter released?")
- Trivial syntax questions with obvious non-existent parameters (e.g., `make_dates=True`)
- Questions where 3 of the 4 choices are visibly absurd

7. Format Requirements for the Output

The output must be pure Markdown formatted exactly as shown below.

- Maintain continuous global numbering from Question 1 to Question 100 across the 10 files:
  - `test-01.md`: Questions 1–10
  - `test-02.md`: Questions 11–20
  - `test-03.md`: Questions 21–30
  - `test-04.md`: Questions 31–40
  - `test-05.md`: Questions 41–50
  - `test-06.md`: Questions 51–60
  - `test-07.md`: Questions 61–70
  - `test-08.md`: Questions 71–80
  - `test-09.md`: Questions 81–90
  - `test-10.md`: Questions 91–100

Format per question:

```markdown
### Question [Number]

[Question context, code block, or scenario]

- [ ] A) [Option text]
- [ ] B) [Option text]
- [ ] C) [Option text]
- [ ] D) [Option text]

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** [Option Letter]) [Option text]

**Explanation:**

- **A is correct/incorrect:** [Detailed rationale]
- **B is correct/incorrect:** [Detailed rationale]
- **C is correct/incorrect:** [Detailed rationale]
- **D is correct/incorrect:** [Detailed rationale]

</details>
```

8. Detailed Explanation Requirements

The explanation must break down all four options:

- Explicitly state why the correct choice is accurate according to the Matplotlib temporal object model
- Detail why each incorrect option fails (e.g., categorical string confusion, invalid format directive, twin-axis scale conflict, or misaligned anchor)
- Include concrete references to Matplotlib mechanics (`mdates.DateFormatter` token rules, `autofmt_xdate` internal logic, `twinx` handle merging, etc.)

9. Distribution Across the 10 Files

- `test-01.md` (Questions 1–10): Temporal Data Types in Matplotlib (Pandas `DatetimeIndex`, Python `datetime`, `date2num`, string pitfalls)
- `test-02.md` (Questions 11–20): Major & Minor Date Locators (`YearLocator`, `MonthLocator`, `WeekdayLocator`, `DayLocator`)
- `test-03.md` (Questions 21–30): Date Formatters & Tokens (`%Y`, `%m`, `%B`, `%b`, `%d`, `DateFormatter`, `ConciseDateFormatter`)
- `test-04.md` (Questions 31–40): Overlap Prevention & Aesthetics (`fig.autofmt_xdate`, label rotation, `ha='right'`, date grids)
- `test-05.md` (Questions 41–50): Temporal Range Locking & Slicing (`set_xlim`, timestamp limits, dynamic zoom windows)
- `test-06.md` (Questions 51–60): Dual-Axis Temporal Plots (`ax.twinx()`, shared date rulers, divergent y-scales, unified legends)
- `test-07.md` (Questions 61–70): Smoothing, Moving Averages & Rolling Statistics (rolling windows, noisy series overlays)
- `test-08.md` (Questions 71–80): Uncertainty Envelopes & Forecast Intervals (`fill_between`, confidence bounds, actual vs. forecast)
- `test-09.md` (Questions 81–90): Event Annotation & Regime Shifts (`axvspan`, `axvline`, anomaly highlights, timestamp text arrows)
- `test-10.md` (Questions 91–100): Machine Learning Time Series Pipelines (train/test temporal splits, residual diagnostics, multi-model forecasts)

10. Quality Control Checklist (14-Point Pre-Submission Gate)

Before finalizing, verify:

1. Exact file naming: `basics/3. Mathplotlib/timeseries/test-01.md` through `test-10.md`
2. Exactly 10 questions per file
3. Continuous numbering 1 through 100
4. Markdown task checkboxes are blank `- [ ]`
5. HTML tags `<details>` and `<summary>` are properly opened and closed
6. Answer key distribution across 100 questions is balanced (~25% each for A, B, C, D)
7. All code snippets are syntactically valid Python 3 / Pandas / Matplotlib
8. Date formatting directives (`%b`, `%m`, `%d`, `%Y`) and locator math are 100% verified
9. Distinction between categorical strings and true `datetime` types is rigorously tested
10. `twinx()` mechanics and unified legend creation are accurately represented
11. `fig.autofmt_xdate()` behavior is properly evaluated
12. Every option (A, B, C, D) has an individual explanation in the accordion
13. No question is a trivial paraphrase of `test.md`
14. No duplicate questions across files

15. Final Deliverables

Generate the 10 files in `basics/3. Mathplotlib/timeseries/` following this prompt.
