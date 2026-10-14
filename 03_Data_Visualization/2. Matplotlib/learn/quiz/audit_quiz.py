import re
import os

def audit_quiz_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Find question blocks
    q_matches = list(re.finditer(r"## Question (\d+) — (Beginner|Intermediate|Advanced|Expert / Advanced)", text))
    print(f"Total question headers found: {len(q_matches)}")
    assert len(q_matches) == 100, f"Expected 100 headers, found {len(q_matches)}"

    # Check numbering continuity
    for idx, match in enumerate(q_matches, 1):
        q_num = int(match.group(1))
        diff = match.group(2)
        assert q_num == idx, f"Header index mismatch: expected {idx}, got {q_num}"
        if 1 <= idx <= 25:
            assert diff == "Beginner", f"Q{idx} difficulty expected Beginner, got {diff}"
        elif 26 <= idx <= 50:
            assert diff == "Intermediate", f"Q{idx} difficulty expected Intermediate, got {diff}"
        elif 51 <= idx <= 75:
            assert diff == "Advanced", f"Q{idx} difficulty expected Advanced, got {diff}"
        elif 76 <= idx <= 100:
            assert diff == "Expert / Advanced", f"Q{idx} difficulty expected Expert / Advanced, got {diff}"

    # Split into question sections
    sections = text.split("## Question ")[1:]
    
    single_count = 0
    multi_count = 0

    for i, sec in enumerate(sections, 1):
        sec_text = "## Question " + sec
        
        # Check topic
        assert "**Topic:**" in sec_text, f"Q{i} missing topic"
        
        # Check type
        if "**Type:** Single Answer" in sec_text:
            single_count += 1
            assert "**Correct Answer:**" in sec_text, f"Q{i} missing single correct answer"
            assert "### Why this is correct" in sec_text, f"Q{i} missing Why this is correct"
            assert "### Why the other options are incorrect" in sec_text, f"Q{i} missing Why the other options are incorrect"
        elif "**Type:** Multi Answer — Select all that apply" in sec_text:
            multi_count += 1
            assert "**Correct Answers:**" in sec_text, f"Q{i} missing multi correct answers"
            assert "### Why these answers are correct" in sec_text, f"Q{i} missing Why these answers are correct"
            assert ("### Why the other option is incorrect" in sec_text or "### Why the other options are incorrect" in sec_text), f"Q{i} missing Why other options are incorrect"
        else:
            raise AssertionError(f"Q{i} has unrecognized or missing Type")

        # Check details tag
        assert "<details>" in sec_text and "</details>" in sec_text, f"Q{i} missing <details> tag"
        assert "<summary>Show Answer & Explanation</summary>" in sec_text, f"Q{i} missing summary tag"

        # Check options
        assert "- A." in sec_text, f"Q{i} missing option A"
        assert "- B." in sec_text, f"Q{i} missing option B"
        assert "- C." in sec_text, f"Q{i} missing option C"
        assert "- D." in sec_text, f"Q{i} missing option D"

        # Ensure correct answer is ONLY inside details
        pre_details, post_details = sec_text.split("<details>", 1)
        assert "**Correct Answer" not in pre_details, f"Q{i} leaks correct answer outside <details>!"
        assert "Why this is correct" not in pre_details, f"Q{i} leaks explanation outside <details>!"
        assert "Why these answers are correct" not in pre_details, f"Q{i} leaks explanation outside <details>!"

    print(f"Audit Complete! Single: {single_count}, Multi: {multi_count}. All 100 questions perfectly validated!")

if __name__ == "__main__":
    audit_quiz_markdown(os.path.join(os.path.dirname(__file__), "matplotlib_quiz.md"))
