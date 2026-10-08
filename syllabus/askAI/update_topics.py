"""
askAI Topic Prompts Sync Utility
--------------------------------
Run this script whenever you update or add markdown files (.md) in the askAI folder.
It scans all .md prompt files across subfolders and updates askAI/js/topics.js automatically.
"""

import os
import json
import sys

# Ensure UTF-8 output formatting for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOPICS_JS_PATH = os.path.join(BASE_DIR, "js", "topics.js")

# Explicit mapping of topic ID -> (folder, list of candidate filenames)
TOPIC_FILE_MAP = {
    "numpy": ("01_Data_Science_Foundations", ["numpy.md"]),
    "pandas": ("01_Data_Science_Foundations", ["pandas.md"]),
    "eda": ("01_Data_Science_Foundations", ["eda.md"]),
    "sqlite": ("01_Data_Science_Foundations", ["sqlite.md"]),
    "scikit-learn": ("02_Machine_Learning", ["scikit-learn.md", "sklearn.md"]),
    "pytorch": ("02_Machine_Learning", ["pytorch.md"]),
    "deep_learning": ("02_Machine_Learning", ["deep_learning.md"]),
    "matplotlib": ("03_Data_Visualization", ["matplotlib.md"]),
    "seaborn": ("03_Data_Visualization", ["seaborn.md"]),
    "plotly": ("03_Data_Visualization", ["plotly.md"]),
    "data_visualization": ("03_Data_Visualization", ["data_visualization.md"]),
    "llm": ("04_Generative_AI_and_LLMs", ["llm.md"]),
    "transformers": ("04_Generative_AI_and_LLMs", ["transformers.md"]),
    "embeddings": ("04_Generative_AI_and_LLMs", ["embeddings.md"]),
    "vector_database": ("04_Generative_AI_and_LLMs", ["vector_database.md"]),
    "chromadb": ("04_Generative_AI_and_LLMs", ["chromadb.md"]),
    "rag": ("04_Generative_AI_and_LLMs", ["rag.md"]),
    "prompt_engineering": ("04_Generative_AI_and_LLMs", ["prompt_engineering.md"]),
    "langchain": ("04_Generative_AI_and_LLMs", ["langchain.md"]),
    "langgraph": ("04_Generative_AI_and_LLMs", ["langgraph.md"]),
    "langsmith": ("04_Generative_AI_and_LLMs", ["langsmith.md"]),
    "pydantic": ("04_Generative_AI_and_LLMs", ["pydantic.md"]),
    "openai_api": ("04_Generative_AI_and_LLMs", ["openai_api.md", "openai.md"]),
    "claude_api": ("04_Generative_AI_and_LLMs", ["claude_api.md", "claude.md"]),
    "fastapi": ("05_Deployment", ["fas.md", "fastapi.md"]),
    "docker": ("05_Deployment", ["docker.md"]),
}


def read_md_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Strip footer if present
    lines = content.splitlines()
    footer_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("**How to use it:**"):
            footer_idx = i
            break

    if footer_idx is not None:
        return "\n".join(lines[:footer_idx]).strip()
    return content.strip()


def sync_prompts():
    if not os.path.exists(TOPICS_JS_PATH):
        print(f"❌ Error: {TOPICS_JS_PATH} not found!")
        return

    with open(TOPICS_JS_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract TOPICS JSON
    idx = text.find("const TOPICS =")
    end = text.find(";\n\nconst FOLDER_LABELS", idx)

    if idx == -1 or end == -1:
        print("❌ Error: Could not locate TOPICS array in js/topics.js!")
        return

    topics_json_str = text[idx + len("const TOPICS = ") : end]
    topics = json.loads(topics_json_str)

    updated_count = 0
    print("🔄 Syncing master prompt files into askAI/js/topics.js...\n")

    for t in topics:
        tid = t["id"]
        if tid in TOPIC_FILE_MAP:
            folder, candidates = TOPIC_FILE_MAP[tid]
            found_path = None
            found_fname = None

            for fname in candidates:
                p = os.path.join(BASE_DIR, folder, fname)
                if os.path.exists(p):
                    found_path = p
                    found_fname = fname
                    break

            if found_path:
                md_text = read_md_file(found_path)
                t["prompt"] = md_text
                updated_count += 1
                print(f"  ✓ [{tid:20s}] <- {folder}/{found_fname} ({len(md_text):,d} chars)")
            else:
                print(f"  ⚠️ [{tid:20s}] No markdown file found in {folder}/ ({candidates})")

    # Preserve FOLDER_LABELS
    folder_idx = text.find("const FOLDER_LABELS =")
    folder_code = text[folder_idx:]

    new_topics_json = json.dumps(topics, ensure_ascii=False, indent=2)
    new_content = f"// ─── TOPICS & FOLDERS DATA ───────────────────────────────────────────\nconst TOPICS = {new_topics_json};\n\n{folder_code}"

    with open(TOPICS_JS_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"\n✅ Successfully updated {updated_count}/{len(topics)} topic prompts in askAI/js/topics.js!")


if __name__ == "__main__":
    sync_prompts()
