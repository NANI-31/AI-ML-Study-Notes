#!/usr/bin/env python3
"""
=============================================================================
LangGraph Study Notes Automated Generator & Quota Sentinel
=============================================================================
Automates the generation of in-depth study notes across all roadmap sections.
Includes quota circuit breaker to halt cleanly when account quota headroom is low.

Features:
- Scans all section markdown files in natural numerical order (1.1, 1.2 ... 34.4).
- Checks quota before starting each topic (via QuotaDashboard / ~/.gemini/quota_accounts).
- Halts gracefully after finishing the current topic if remaining quota < threshold.
- Records state to generation_progress.json for seamless resumption.
- Atomic file writes (writes to .tmp first to prevent file corruption).
- Zero third-party dependencies (pure Python standard library).
"""

import os
import sys
import json
import time
import re
import glob
import signal
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

# Ensure clean UTF-8 output across Windows PowerShell / CMD consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Default Paths & Configuration
WORKSPACE_ROOT = Path(__file__).resolve().parent
SYLLABUS_DIR = WORKSPACE_ROOT
PROGRESS_FILE = SYLLABUS_DIR / "generation_progress.json"

USER_PROFILE = os.environ.get("USERPROFILE", "C:\\Users\\Chowd")
ACCOUNTS_DIR = Path(USER_PROFILE) / ".gemini" / "quota_accounts"
ORDER_FILE = ACCOUNTS_DIR / "order.json"

# Quota Threshold: Halt if remaining fraction is less than this (e.g. 0.15 = 15%)
DEFAULT_MIN_QUOTA = 0.15
MIN_FILE_SIZE_COMPLETED = 5000  # Bytes

# Interruption flag
ABORT_REQUESTED = False

def handle_sigint(sig, frame):
    global ABORT_REQUESTED
    print("\n[!] Ctrl+C detected. Finishing current step and halting safely...")
    ABORT_REQUESTED = True

signal.signal(signal.SIGINT, handle_sigint)

# ---------------------------------------------------------------------------
# Quota Inspection & Sentinel
# ---------------------------------------------------------------------------

def query_local_quota_daemon(port=3210):
    """Attempt to fetch live quota from the local QuotaDashboard daemon."""
    url = f"http://127.0.0.1:{port}/api/status"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NotesGenerator/1.0"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data, None
    except Exception as e:
        return None, str(e)
    return None, "Offline"

def get_active_account_info():
    """Retrieve active account details and quota cache from ~/.gemini/quota_accounts/."""
    # 1. Try local daemon first
    daemon_data, _ = query_local_quota_daemon()
    if daemon_data and "accounts" in daemon_data:
        active_acc = next((a for a in daemon_data["accounts"] if a.get("isActive")), daemon_data["accounts"][0])
        return {
            "source": "quota_dashboard_daemon",
            "account_id": active_acc.get("fileId", "active"),
            "account_name": active_acc.get("name") or active_acc.get("email") or "Active Account",
            "email": active_acc.get("email", ""),
            "quotas": active_acc.get("quotas", {})
        }, None

    # 2. Fallback to reading disk cache
    if not ORDER_FILE.exists():
        return None, f"Order file not found at {ORDER_FILE}"

    try:
        with open(ORDER_FILE, "r", encoding="utf-8") as f:
            order = json.load(f)
        if not order:
            return None, "order.json is empty"
        active_acc_id = order[0]

        quota_file = ACCOUNTS_DIR / f"{active_acc_id}_quota.json"
        acc_file = ACCOUNTS_DIR / f"{active_acc_id}.json"

        acc_name = active_acc_id
        acc_email = ""
        if acc_file.exists():
            try:
                with open(acc_file, "r", encoding="utf-8") as af:
                    adata = json.load(af)
                    acc_name = adata.get("name") or adata.get("email") or active_acc_id
                    acc_email = adata.get("email", "")
            except Exception:
                pass

        if not quota_file.exists():
            return None, f"Quota cache {quota_file.name} not found"

        with open(quota_file, "r", encoding="utf-8") as qf:
            quota_data = json.load(qf)

        return {
            "source": "disk_cache",
            "account_id": active_acc_id,
            "account_name": acc_name,
            "email": acc_email,
            "quotas": quota_data
        }, None
    except Exception as e:
        return None, str(e)

def check_quota_headroom(model_type="gemini", min_quota=DEFAULT_MIN_QUOTA):
    """
    Checks whether the active account has enough quota remaining.
    Returns: (is_ok, remaining_fraction, reset_time, description)
    """
    info, err = get_active_account_info()
    if err or not info:
        # If quota monitor is offline or credentials unavailable, allow pass-through with warning
        return True, 1.0, "Unknown", f"Quota monitor warning: {err or 'no info'}"

    quotas = info.get("quotas", {})
    groups = quotas.get("groups", [])
    if not groups:
        return True, 1.0, "Unknown", "No quota groups found"

    target_group = None
    for g in groups:
        dname = g.get("displayName", "").lower()
        if model_type == "gemini" and "gemini" in dname:
            target_group = g
            break
        elif model_type in ["claude", "gpt"] and ("claude" in dname or "gpt" in dname):
            target_group = g
            break

    if not target_group:
        target_group = groups[0]

    buckets = target_group.get("buckets", [])
    # Check 5h sprint bucket first, then weekly bucket
    bucket_5h = next((b for b in buckets if b.get("window") == "5h"), None)
    bucket_weekly = next((b for b in buckets if b.get("window") == "weekly"), None)

    evaluated_bucket = bucket_5h or bucket_weekly or (buckets[0] if buckets else None)
    if not evaluated_bucket:
        return True, 1.0, "Unknown", "No bucket found"

    remaining = evaluated_bucket.get("remainingFraction", 1.0)
    reset_time = evaluated_bucket.get("resetTime", "N/A")
    desc = f"{target_group.get('displayName')} [{evaluated_bucket.get('window')}]: {remaining*100:.1f}% remaining (Reset: {reset_time})"

    is_ok = remaining >= min_quota
    return is_ok, remaining, reset_time, desc

# ---------------------------------------------------------------------------
# File Queue Discovery & Sorting
# ---------------------------------------------------------------------------

def get_natural_sort_key(file_path):
    """Sort files numerically (e.g. 1.1, 1.2 ... 3.1, 3.2 ... 10.1 ... 34.4)."""
    fname = Path(file_path).stem
    m = re.match(r"^(\d+)\.(\d+)$", fname)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (999, 999)

def discover_study_files(syllabus_dir=SYLLABUS_DIR):
    """Scan all section markdown files and categorize into completed and pending."""
    pattern = str(syllabus_dir / "*" / "*.md")
    files = glob.glob(pattern)
    files.sort(key=get_natural_sort_key)

    completed = []
    pending = []

    for f in files:
        p = Path(f)
        size = p.stat().st_size
        if size >= MIN_FILE_SIZE_COMPLETED:
            completed.append({"path": p, "size": size, "name": p.name})
        else:
            pending.append({"path": p, "size": size, "name": p.name})

    return completed, pending

# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

SYSTEM_DIRECTIVE = """You are a Principal AI/ML Staff Engineer and expert educator in Agentic AI, Generative AI, and LangGraph.
Produce an exhaustive, production-grade, authoritative study guide for the requested syllabus topic.
Follow these strict structural requirements:

1. Top-Level Architectural Overview:
   - Provide a deep conceptual introduction explaining theoretical and computational foundations.
   - Include clear text/ASCII or Mermaid architectural diagrams illustrating the data-flow, state transitions, and component interactions.

2. Table of Contents:
   - Detailed, clickable markdown TOC referencing every numbered topic and subtopic.

3. Deep Technical Breakdown of Every Point:
   - Address every single numbered subtopic in order with extreme technical depth.
   - Theoretical rigor + real-world production gotchas and edge cases.
   - Modern LangGraph code examples (using modern idioms: StateGraph, START, END, typing Annotated, TypedDict/Pydantic state schemas, channel reducers, Send API, Command API, and RunnableConfig).
   - All code must be clean, runnable, well-commented, and production-grade.

4. Comprehensive Interview Q&A Section:
   - Include 15 to 25 senior/staff level interview questions and comprehensive, authoritative answers.
   - Cover architectural trade-offs, concurrency issues, edge-case failure modes, checkpoint persistence, and distributed state.

5. Master Study Checklist & Key Takeaways:
   - Provide a clean ASCII box checklist summarizing the essential production rules, gotchas, and invariants.
"""

def build_prompt_from_file(file_path):
    """Reads the skeleton topic file and constructs the comprehensive user prompt."""
    with open(file_path, "r", encoding="utf-8") as f:
        skeleton_content = f.read().strip()

    user_prompt = f"""i am learning the ai/ml, gen ai course. now i will give you the part of the road map of the LangGraph. so give me the detailed description with examples and include interview questions and answers at end of response.

{skeleton_content}
"""
    return user_prompt

# ---------------------------------------------------------------------------
# LLM Generation Clients
# ---------------------------------------------------------------------------

def call_gemini_api(prompt, api_key, model="gemini-2.0-flash"):
    """Generate notes using Google Gemini REST API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{SYSTEM_DIRECTIVE}\n\n---\n\n{prompt}"}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8192
        }
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=180) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        candidates = res_data.get("candidates", [])
        if candidates and "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            return text
        raise RuntimeError(f"Gemini API returned no candidates: {res_data}")

def call_openai_compatible_api(prompt, api_key, base_url="https://api.openai.com/v1", model="gpt-4o"):
    """Generate notes using an OpenAI-compatible endpoint (OpenAI, Groq, DeepSeek, Ollama)."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_DIRECTIVE},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )

    with urllib.request.urlopen(req, timeout=180) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        choices = res_data.get("choices", [])
        if choices and "message" in choices[0]:
            return choices[0]["message"].get("content", "")
        raise RuntimeError(f"API returned unexpected format: {res_data}")

# ---------------------------------------------------------------------------
# Atomic Write & State Persistence
# ---------------------------------------------------------------------------

def write_markdown_atomic(target_path, content):
    """Write content to a temporary file, then atomically replace target."""
    tmp_path = target_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    # Atomic rename / replace
    os.replace(tmp_path, target_path)

def load_progress():
    data = {"completed": [], "last_run": None}
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    data.update(loaded)
        except Exception:
            pass
    if "completed" not in data or not isinstance(data["completed"], list):
        data["completed"] = []
    return data

def save_progress(progress_data):
    progress_data["last_run"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, indent=2)

# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Automated LangGraph Study Notes Generator with Quota Sentinel")
    parser.add_argument("--dry-run", action="store_true", help="Scan queue, check quota, and print status without calling LLM")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of topics to process in this run (0 = unlimited)")
    parser.add_argument("--min-quota", type=float, default=DEFAULT_MIN_QUOTA, help="Minimum remaining quota fraction before graceful halt (e.g. 0.15 = 15%)")
    parser.add_argument("--provider", choices=["gemini", "openai", "custom"], default="gemini", help="LLM Provider")
    parser.add_argument("--model", type=str, default="", help="Model name (default: gemini-2.0-flash or gpt-4o)")
    parser.add_argument("--api-key", type=str, default="", help="API key (defaults to GEMINI_API_KEY or OPENAI_API_KEY from env)")
    parser.add_argument("--base-url", type=str, default="", help="Base URL for OpenAI-compatible endpoint")
    parser.add_argument("--auto-wait", action="store_true", help="Automatically sleep until quota reset if quota is low")

    args = parser.parse_args()

    # Model defaults
    if not args.model:
        args.model = "gemini-2.0-flash" if args.provider == "gemini" else "gpt-4o"

    # API key resolution
    api_key = args.api_key
    if not api_key:
        if args.provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY", "")
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")

    print("\n" + "="*70)
    print("🚀 LANGGRAPH STUDY NOTES AUTOMATED GENERATOR & QUOTA SENTINEL")
    print("="*70)

    # 1. Discover Queue
    completed, pending = discover_study_files(SYLLABUS_DIR)
    print(f"📁 Syllabus Root    : {SYLLABUS_DIR}")
    print(f"📊 Total Sections   : {len(completed) + len(pending)}")
    print(f"✅ Already Completed: {len(completed)}")
    print(f"⏳ Pending Queue    : {len(pending)}")

    # 2. Check Quota Sentinel
    model_type = "gemini" if "gemini" in args.model.lower() else "claude"
    is_ok, rem_pct, reset_time, quota_desc = check_quota_headroom(model_type, args.min_quota)
    print(f"🛡️  Quota Status     : {quota_desc}")

    if not is_ok:
        print("\n" + "!"*70)
        print(f"⚠️  QUOTA CIRCUIT BREAKER: Headroom ({rem_pct*100:.1f}%) is below minimum threshold ({args.min_quota*100:.1f}%).")
        print(f"   Quota Reset Time: {reset_time}")
        print("   Halting execution before starting any task to protect account headroom.")
        print("!"*70 + "\n")
        sys.exit(0)

    # 3. Dry-run Mode
    if args.dry_run:
        print("\n[DRY RUN] Next 10 pending topics:")
        for i, item in enumerate(pending[:10], 1):
            rel_path = item["path"].relative_to(SYLLABUS_DIR)
            print(f"  {i:2d}. {rel_path} (current size: {item['size']} B)")
        print("\nDry run completed. To run generation, provide an API key or run without --dry-run.")
        sys.exit(0)

    if not api_key and not args.dry_run:
        print("\n❌ Error: No API key found.")
        print(f"   Please set the GEMINI_API_KEY or OPENAI_API_KEY environment variable,")
        print(f"   or pass --api-key <YOUR_KEY>.")
        print("   Tip: You can get a free Gemini API key at: https://aistudio.google.com/")
        sys.exit(1)

    # 4. Processing Loop
    processed_in_this_run = 0
    progress = load_progress()

    for item in pending:
        if ABORT_REQUESTED:
            print("\n[!] Gracefully stopping pipeline as requested.")
            break

        if args.limit > 0 and processed_in_this_run >= args.limit:
            print(f"\n🎯 Reached batch limit of {args.limit} tasks. Stopping run.")
            break

        # Check Quota Sentinel BEFORE starting next task
        is_ok, rem_pct, reset_time, quota_desc = check_quota_headroom(model_type, args.min_quota)
        if not is_ok:
            print("\n" + "="*70)
            print(f"⏸️  QUOTA LIMIT LOW ({rem_pct*100:.1f}% remaining < {args.min_quota*100:.1f}% threshold).")
            print(f"   Previous task was cleanly saved.")
            print(f"   Next task in queue: {item['path'].name}")
            print(f"   Quota resets at   : {reset_time}")
            print("="*70 + "\n")
            save_progress(progress)
            break

        file_path = item["path"]
        rel_path = file_path.relative_to(SYLLABUS_DIR)
        print(f"\n▶ [{processed_in_this_run + 1}] Processing: {rel_path}...")

        start_time = time.time()
        prompt = build_prompt_from_file(file_path)

        try:
            if args.provider == "gemini":
                generated_text = call_gemini_api(prompt, api_key, args.model)
            else:
                base_url = args.base_url or "https://api.openai.com/v1"
                generated_text = call_openai_compatible_api(prompt, api_key, base_url, args.model)

            # Atomic save
            write_markdown_atomic(file_path, generated_text)
            duration = time.time() - start_time
            file_size_kb = file_path.stat().st_size / 1024

            print(f"   ✅ Successfully wrote {file_path.name} ({file_size_kb:.1f} KB in {duration:.1f}s)")

            # Record in progress ledger
            progress["completed"].append({
                "file": str(rel_path),
                "timestamp": datetime.now().isoformat(),
                "size_kb": round(file_size_kb, 1),
                "duration_s": round(duration, 1)
            })
            save_progress(progress)

            processed_in_this_run += 1

            # Brief politeness pause between API calls (1.5 seconds)
            time.sleep(1.5)

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            print(f"\n❌ HTTP Error {e.code}: {e.reason}")
            if e.code == 429 or "RESOURCE_EXHAUSTED" in err_body:
                print("⚠️ Rate limit or quota exhausted! Pausing pipeline.")
                save_progress(progress)
                break
            else:
                print(f"Response: {err_body[:300]}")
                break
        except Exception as e:
            print(f"\n❌ Error processing {file_path.name}: {e}")
            save_progress(progress)
            break

    print("\n" + "="*70)
    print(f"🏁 RUN FINISHED: Generated {processed_in_this_run} notes in this session.")
    print(f"📊 Remaining pending: {len(pending) - processed_in_this_run}")
    print("="*70)

if __name__ == "__main__":
    main()
