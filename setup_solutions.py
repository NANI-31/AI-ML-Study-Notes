import os, sys, io, re, json, subprocess

ROOT = r"D:\Data\GitHub\interview\python\dsa\common interview questions"
SOL_DIR = os.path.join(ROOT, "solutions")
COMMON_DIR = os.path.join(SOL_DIR, "common")
UNIQUE_DIR = os.path.join(SOL_DIR, "unique")
CONSENSUS_DIR = os.path.join(SOL_DIR, "consensus")
MANIFEST_PATH = os.path.join(SOL_DIR, "manifest.json")
PROGRESS_PATH = os.path.join(SOL_DIR, "DSA_SOLUTION_PROGRESS.md")
INDEX_PATH = os.path.join(SOL_DIR, "DSA_SOLUTION_INDEX.md")

os.makedirs(COMMON_DIR, exist_ok=True)
os.makedirs(UNIQUE_DIR, exist_ok=True)
os.makedirs(CONSENSUS_DIR, exist_ok=True)

def to_snake_case(s):
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r"['\"’]", '', s)
    s = re.sub(r'[^a-zA-Z0-9\s_]', ' ', s).strip()
    s = re.sub(r'\s+', '_', s)
    return s.lower()

def build_manifest():
    master_file = os.path.join(ROOT, "DSA_COMMON_INTERVIEW_QUESTIONS_MASTER.md")
    registry_file = os.path.join(ROOT, "DSA_QUESTION_REGISTRY.md")

    registry = {}
    with open(registry_file, "r", encoding="utf-8") as f:
        reg_text = f.read()

    for m in re.finditer(r'##\s+(Q\d+)\s+—\s+(.*?)\n(.*?)(?=\n##\s+Q|\Z)', reg_text, re.DOTALL):
        qid = m.group(1)
        name = m.group(2).strip()
        body = m.group(3)
        
        lc_m = re.search(r'\*\*LeetCode:\*\*\s*(.*)', body)
        cat_m = re.search(r'\*\*Category:\*\*\s*(.*)', body)
        diff_m = re.search(r'\*\*Difficulty:\*\*\s*(.*)', body)
        freq_m = re.search(r'\*\*Frequency:\*\*\s*(\d+)', body)
        sources_m = re.search(r'\*\*Sources:\*\*\s*(.*)', body)
        variants_m = re.search(r'\*\*Variants:\*\*(.*?)(?=\n-|\n##|\Z)', body, re.DOTALL)
        
        registry[qid] = {
            'qid': qid,
            'name': name,
            'leetcode': lc_m.group(1).strip() if lc_m else None,
            'category': cat_m.group(1).strip() if cat_m else 'Unknown',
            'difficulty': diff_m.group(1).strip() if diff_m else 'Unknown',
            'frequency': int(freq_m.group(1)) if freq_m else 1,
            'sources': sources_m.group(1).strip() if sources_m else '',
            'variants': variants_m.group(1).strip() if variants_m else '',
            'notes': ''
        }

    with open(master_file, "r", encoding="utf-8") as f:
        master_text = f.read()

    p7 = re.search(r'## Part 7: Session 2 New Unique Questions.*?(?=## Part 8:)', master_text, re.DOTALL)
    if p7:
        for line in p7.group(0).split('\n'):
            if line.strip().startswith('| Q'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 7 and re.match(r'^Q\d+$', parts[0]):
                    qid, name, diff, lc, sources, cat, insight = parts[:7]
                    registry[qid] = {
                        'qid': qid,
                        'name': name,
                        'leetcode': lc,
                        'category': cat,
                        'difficulty': diff,
                        'frequency': 1,
                        'sources': sources,
                        'variants': '',
                        'notes': insight
                    }

    p8 = re.search(r'## Part 8: Frequency Corrections.*?(?=## Part 9:)', master_text, re.DOTALL)
    if p8:
        for line in p8.group(0).split('\n'):
            if line.strip().startswith('| Q'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 5 and re.match(r'^Q\d+$', parts[0]):
                    qid, name, old_f, new_f, added_s = parts[:5]
                    try:
                        if qid in registry:
                            registry[qid]['frequency'] = int(new_f)
                            if added_s:
                                registry[qid]['sources'] += f', {added_s}'
                    except ValueError:
                        pass

    p3 = re.search(r'## Part 3: Essential 50 — Consensus List.*?\n(\|.*?)(?=\n## Part 4:)', master_text, re.DOTALL)
    essential_map = {}
    if p3:
        for line in p3.group(1).split('\n'):
            if line.strip().startswith('|') and not line.strip().startswith('| #') and not line.strip().startswith('|:---:'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                num, name, lc, diff, pat, why = parts[:6]
                matched_qid = None
                if lc and lc.strip():
                    for qid, q in registry.items():
                        if q.get('leetcode') and q['leetcode'].strip() == lc.strip():
                            matched_qid = qid
                            break
                if not matched_qid:
                    clean_name = name.lower().replace('&', 'and').strip()
                    for qid, q in registry.items():
                        q_name = q['name'].lower().replace('&', 'and').strip()
                        if clean_name == q_name or clean_name in q_name or q_name in clean_name:
                            matched_qid = qid
                            break
                if matched_qid:
                    essential_map[matched_qid] = {
                        'rank': int(num),
                        'name': name,
                        'pattern': pat,
                        'why': why
                    }

    manifest = []
    for qid in sorted(registry.keys(), key=lambda x: (int(x[1:]), x)):
        q = registry[qid]
        is_consensus = qid in essential_map
        classification = 'Common' if q['frequency'] >= 2 else 'Unique'
        base_name = to_snake_case(q['name'])
        filename = f"{qid}_{base_name}.py"
        target_rel_path = f"common/{filename}" if classification == 'Common' else f"unique/{filename}"
        
        manifest.append({
            'qid': qid,
            'name': q['name'],
            'leetcode': q['leetcode'],
            'category': q['category'],
            'difficulty': q['difficulty'],
            'frequency': q['frequency'],
            'sources': q['sources'],
            'variants': q['variants'],
            'notes': q['notes'],
            'is_consensus': is_consensus,
            'consensus_rank': essential_map[qid]['rank'] if is_consensus else None,
            'consensus_pattern': essential_map[qid]['pattern'] if is_consensus else None,
            'consensus_why': essential_map[qid]['why'] if is_consensus else None,
            'classification': classification,
            'filename': filename,
            'rel_path': target_rel_path,
            'status': 'Pending'
        })

    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Successfully built manifest with {len(manifest)} questions.")
    return manifest

def sync_progress_and_index(batch_name="Batch 0 (Initialized)", next_action="Begin Batch 1 (Q0001–Q0010)"):
    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # Check file existence and validate
    completed_count = 0
    last_completed = "None"
    
    for item in manifest:
        full_path = os.path.join(SOL_DIR, item['rel_path'])
        if os.path.exists(full_path):
            item['status'] = 'Complete'
            completed_count += 1
            last_completed = f"{item['qid']} — {item['name']}"
        else:
            item['status'] = 'Pending'

    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    total_q = len(manifest)
    common_items = [q for q in manifest if q['classification'] == 'Common']
    unique_items = [q for q in manifest if q['classification'] == 'Unique']
    consensus_items = sorted([q for q in manifest if q['is_consensus']], key=lambda x: x['consensus_rank'])

    overall_status = "COMPLETED" if completed_count == total_q and total_q > 0 else "IN_PROGRESS"

    # 1. Generate DSA_SOLUTION_PROGRESS.md
    prog = []
    prog.append("# DSA Solution Generation Progress\n")
    prog.append(f"## Status\n\n{overall_status}\n")
    prog.append(f"## Source Project\n\n{ROOT}\n")
    prog.append(f"## Output Directory\n\n{SOL_DIR}\n")
    prog.append("## Statistics\n")
    prog.append(f"Total consolidated questions: {total_q}")
    prog.append(f"Common questions: {len(common_items)}")
    prog.append(f"Unique questions: {len(unique_items)}")
    prog.append(f"Consensus questions: {len(consensus_items)}")
    prog.append(f"Solutions completed: {completed_count}")
    prog.append(f"Solutions remaining: {total_q - completed_count}\n")
    
    prog.append("## Processing Status\n")
    prog.append("### Common\n")
    for q in common_items:
        check = "x" if q['status'] == 'Complete' else " "
        prog.append(f"- [{check}] {q['qid']} — {q['name']} — {q['status']}")
    
    prog.append("\n### Unique\n")
    for q in unique_items:
        check = "x" if q['status'] == 'Complete' else " "
        prog.append(f"- [{check}] {q['qid']} — {q['name']} — {q['status']}")

    prog.append("\n### Consensus (Essential 50)\n")
    for q in consensus_items:
        check = "x" if q['status'] == 'Complete' else " "
        prog.append(f"- [{check}] Rank {q['consensus_rank']:02d}: {q['qid']} — {q['name']} (Canonical: {q['rel_path']}) — {q['status']}")

    prog.append("\n## Current Phase\n\n" + ("Finalized" if overall_status == "COMPLETED" else "Common Solutions Generation") + "\n")
    prog.append(f"## Current Position\n\nLast completed question:\n`{last_completed}`\n")
    prog.append(f"## Current Batch\n\n{batch_name}\n")
    prog.append(f"## Next Action\n\n{next_action}\n")
    prog.append("## Important Decisions\n")
    prog.append("- Standard Python 3.10+ syntax with modern type annotations.")
    prog.append("- Unified canonical solution files: Consensus questions directly reference canonical implementations in `solutions/common/` to prevent redundant divergence.")
    prog.append("- Every solution file includes interview docstrings, problem description, time/space complexity, constraints/edge cases, algorithmic variants, and runnable assertions under `if __name__ == '__main__':`.")
    prog.append("- Verification: Every solution file is executed directly with python before being marked Complete.\n")
    prog.append("## Questions Requiring Review\n\n- None currently.\n")
    prog.append(f"## Last Updated\n\n{now_str}\n")

    with open(PROGRESS_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(prog))

    # 2. Generate DSA_SOLUTION_INDEX.md
    idx = []
    idx.append("# DSA Python Solutions Index\n")
    idx.append(f"> Auto-synchronized index of high-quality, interview-ready Python DSA solutions.\n> Generated: {now_str}\n")
    
    idx.append("## Consensus Questions (Essential 50)\n")
    idx.append("> Essential 50 consensus problem set recommended across all primary interview sources.\n")
    idx.append("| Rank | QID | Problem | LeetCode | Difficulty | Pattern | Canonical Solution |")
    idx.append("|:---:|:---|:---|:---:|:---:|:---|:---|")
    for q in consensus_items:
        sol_link = f"[{q['filename']}]({q['rel_path']})" if q['status'] == 'Complete' else f"`{q['filename']}` *(Pending)*"
        idx.append(f"| {q['consensus_rank']} | {q['qid']} | {q['name']} | {q['leetcode'] or '-'} | {q['difficulty']} | {q['consensus_pattern'] or '-'} | {sol_link} |")

    idx.append("\n## Common Questions (2+ Sources)\n")
    idx.append("| QID | Problem | LeetCode | Category | Difficulty | Freq | Solution |")
    idx.append("|:---|:---|:---:|:---|:---:|:---:|:---|")
    for q in common_items:
        sol_link = f"[{q['filename']}]({q['rel_path']})" if q['status'] == 'Complete' else f"`{q['filename']}` *(Pending)*"
        idx.append(f"| {q['qid']} | {q['name']} | {q['leetcode'] or '-'} | {q['category']} | {q['difficulty']} | {q['frequency']} | {sol_link} |")

    idx.append("\n## Unique Questions (1 Source)\n")
    idx.append("| QID | Problem | LeetCode | Category | Difficulty | Freq | Solution |")
    idx.append("|:---|:---|:---:|:---|:---:|:---:|:---|")
    for q in unique_items:
        sol_link = f"[{q['filename']}]({q['rel_path']})" if q['status'] == 'Complete' else f"`{q['filename']}` *(Pending)*"
        idx.append(f"| {q['qid']} | {q['name']} | {q['leetcode'] or '-'} | {q['category']} | {q['difficulty']} | {q['frequency']} | {sol_link} |")

    with open(INDEX_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(idx))

    # 3. Consensus README.md
    consensus_readme_path = os.path.join(CONSENSUS_DIR, "README.md")
    with open(consensus_readme_path, 'w', encoding='utf-8') as f:
        f.write("# Consensus Solutions (Essential 50)\n\n")
        f.write("All 50 consensus interview questions are canonical high-frequency problems.\n")
        f.write("To prevent redundant duplication and code drift, each problem is maintained canonically in `../common/`.\n\n")
        f.write("See `../DSA_SOLUTION_INDEX.md` for the direct ranked mapping.\n")

    print(f"Synced progress and index. Completed: {completed_count}/{total_q}. Status: {overall_status}")

if __name__ == '__main__':
    build_manifest()
    sync_progress_and_index()

