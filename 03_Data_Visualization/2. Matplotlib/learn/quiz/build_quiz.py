import os
import sys

from part1_beginner import questions_part1
from part2_intermediate import questions_part2
from part3_advanced import questions_part3
from part4_expert import questions_part4

all_questions = questions_part1 + questions_part2 + questions_part3 + questions_part4

def validate_questions(questions):
    assert len(questions) == 100, f"Expected 100 questions, got {len(questions)}"
    
    single_count = 0
    multi_count = 0
    
    for i, q in enumerate(questions, 1):
        assert q["num"] == i, f"Question index mismatch at {i}: got {q['num']}"
        assert "difficulty" in q, f"Missing difficulty in Q{i}"
        assert "topic" in q, f"Missing topic in Q{i}"
        assert "question" in q, f"Missing question text in Q{i}"
        assert "options" in q, f"Missing options in Q{i}"
        assert set(q["options"].keys()) == {"A", "B", "C", "D"} or (len(q["options"]) >= 4), f"Invalid options in Q{i}"
        
        if q["type"] == "single":
            single_count += 1
            assert isinstance(q["correct"], str), f"Q{i} single answer must have string correct answer"
            assert q["correct"] in q["options"], f"Q{i} correct answer not in options"
            assert "correct_explanation" in q, f"Q{i} missing correct_explanation"
            assert "incorrect_explanations" in q, f"Q{i} missing incorrect_explanations"
            incorrect_keys = set(q["options"].keys()) - {q["correct"]}
            for k in incorrect_keys:
                assert k in q["incorrect_explanations"], f"Q{i} missing incorrect explanation for option {k}"
        elif q["type"] == "multi":
            multi_count += 1
            assert isinstance(q["correct"], list), f"Q{i} multi answer must have list of correct answers"
            assert len(q["correct"]) >= 2, f"Q{i} multi answer must have >= 2 correct answers"
            assert len(q["correct"]) <= len(q["options"]), f"Q{i} multi answer cannot exceed total options"
            assert "correct_explanations" in q, f"Q{i} missing correct_explanations"
            assert "incorrect_explanations" in q, f"Q{i} missing incorrect_explanations"
            for k in q["correct"]:
                assert k in q["correct_explanations"], f"Q{i} missing correct explanation for option {k}"
            incorrect_keys = set(q["options"].keys()) - set(q["correct"])
            for k in incorrect_keys:
                assert k in q["incorrect_explanations"], f"Q{i} missing incorrect explanation for option {k}"
        else:
            raise ValueError(f"Unknown question type: {q['type']} in Q{i}")

    print(f"Validation successful! Total: {len(questions)} (Single: {single_count}, Multi: {multi_count})")
    return single_count, multi_count

def render_question_md(q):
    lines = []
    
    # Question Header
    diff = q["difficulty"]
    lines.append(f"## Question {q['num']} — {diff}\n")
    lines.append(f"**Topic:** {q['topic']}  ")
    if q["type"] == "single":
        lines.append("**Type:** Single Answer\n")
    else:
        lines.append("**Type:** Multi Answer — Select all that apply\n")
        
    lines.append(f"{q['question']}\n")
    
    # Options
    for opt_key in sorted(q["options"].keys()):
        lines.append(f"- {opt_key}. {q['options'][opt_key]}")
    lines.append("")
    
    # Hidden Details Section
    lines.append("<details>")
    lines.append("<summary>Show Answer & Explanation</summary>\n")
    
    if q["type"] == "single":
        correct_key = q["correct"]
        correct_text = q["options"][correct_key]
        lines.append(f"**Correct Answer:** {correct_key}. {correct_text}\n")
        lines.append("### Why this is correct\n")
        lines.append(f"{q['correct_explanation']}\n")
        lines.append("### Why the other options are incorrect\n")
        for opt_key in sorted(q["options"].keys()):
            if opt_key != correct_key:
                lines.append(f"- **{opt_key}. {q['options'][opt_key]}:** {q['incorrect_explanations'][opt_key]}")
    else:
        correct_keys = sorted(q["correct"])
        lines.append(f"**Correct Answers:** {', '.join(correct_keys)}\n")
        lines.append("### Why these answers are correct\n")
        for k in correct_keys:
            lines.append(f"- **{k}:** {q['correct_explanations'][k]}")
        
        incorrect_keys = sorted(set(q["options"].keys()) - set(correct_keys))
        if len(incorrect_keys) == 1:
            lines.append("\n### Why the other option is incorrect\n")
            k = incorrect_keys[0]
            lines.append(f"- **{k}:** {q['incorrect_explanations'][k]}")
        elif len(incorrect_keys) > 1:
            lines.append("\n### Why the other options are incorrect\n")
            for k in incorrect_keys:
                lines.append(f"- **{k}:** {q['incorrect_explanations'][k]}")
                
    lines.append("\n</details>\n")
    lines.append("---\n")
    return "\n".join(lines)

def build_full_quiz():
    single_c, multi_c = validate_questions(all_questions)
    
    header = f"""# Comprehensive 100-Question Matplotlib Mastery Quiz

Welcome to the **Comprehensive 100-Question Matplotlib Mastery Quiz**. This self-assessment suite spans the entire spectrum of Matplotlib—from foundational plotting primitives and procedural `pyplot` workflows to intermediate subplots, twin axes, styling, advanced transformations, and low-level backend architectures.

---

## Quiz Instructions

1. **Attempt Each Question First:** Read each question and the provided code snippet carefully before revealing the answer.
2. **Question Types:**
   - **Single Answer:** Select exactly one best answer.
   - **Multi Answer (Select all that apply):** Select all valid options (contains at least two correct options).
3. **Check Answers & Explanations:** Click on the **`Show Answer & Explanation`** dropdown under each question to inspect the detailed technical breakdown for both correct and incorrect options.
4. **Target Difficulty Breakdown:**
   - **Questions 1–25:** Beginner (Foundations, `pyplot`, basic charts, colors, markers, limits)
   - **Questions 26–50:** Intermediate (Object-Oriented API, subplots, twin axes, dates, images, colormaps, layout)
   - **Questions 51–75:** Advanced (GridSpec, custom transforms, polar/3D, spines, patches, custom colormaps)
   - **Questions 76–100:** Expert / Advanced (Backend architecture, Artist hierarchy, blitting, event loops, memory management, custom projections)

---

## Summary Statistics

- **Total Questions:** 100
- **Single-Answer Questions:** {single_c}
- **Multi-Answer Questions:** {multi_c}
- **Verified Matplotlib Version:** 3.11.x (compatible with modern 3.x standards)

---

"""
    
    body = "".join(render_question_md(q) for q in all_questions)
    
    footer = """## Quiz Completion & Next Milestones

Congratulations on completing the **100-Question Matplotlib Mastery Quiz**!

### Scoring Guide
- **90–100%:** **Master / Architect Level** — Exceptional mastery of Matplotlib's object-oriented architecture, custom projection pipeline, and performance optimization.
- **75–89%:** **Senior Visualizer** — Strong understanding of complex multi-axes layouts, custom transformations, and color normalization.
- **60–74%:** **Competent Practitioner** — Solid grasp of everyday data visualization workflows, subplots, and styling.
- **Below 60%:** **Foundational Review Needed** — Review the official Matplotlib documentation, specifically the *Artist Tutorial*, *Transformations Tutorial*, and *Lifecycle of a Plot*.

---
"""

    full_content = header + body + footer
    return full_content

if __name__ == "__main__":
    content = build_full_quiz()
    out_path = os.path.join(os.path.dirname(__file__), "matplotlib_quiz.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully generated full quiz at: {out_path} ({len(content.splitlines())} lines)")
