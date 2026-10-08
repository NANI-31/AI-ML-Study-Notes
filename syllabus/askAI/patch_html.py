#!/usr/bin/env python3
"""
Patch prompt_builder.html to fix the backtick-in-template-literal bug.
The roadmap text (from MD files) contains backticks which break JS template literals.
Fix: remove roadmap from the innerHTML template, set it via textContent afterward.
"""
import sys, re
sys.stdout.reconfigure(encoding='utf-8')

with open('prompt_builder.html', encoding='utf-8') as f:
    content = f.read()

# ── Fix 1: Remove ${t.roadmap} from innerHTML template literal ────────────────
# Replace:   <div class="roadmap-body" id="roadmap-body">${t.roadmap || "(no roadmap found)"}</div>
# With:      <div class="roadmap-body" id="roadmap-body"></div>
old_rm = '<div class="roadmap-body" id="roadmap-body">${t.roadmap || "(no roadmap found)"}</div>'
new_rm = '<div class="roadmap-body" id="roadmap-body"></div>'
if old_rm in content:
    content = content.replace(old_rm, new_rm)
    print("Fix 1: roadmap removed from template literal  OK")
else:
    print("Fix 1: pattern NOT found")
    # show what's around roadmap-body
    idx = content.find('roadmap-body')
    print("  Context:", repr(content[idx:idx+200]))

# ── Fix 2: Also remove ${t.icon} and ${t.label} — both can contain backticks
# Wait — actually icons are emoji, not backtick. Labels are plain text. Those are fine.
# Only roadmap and prompt content have backticks.

# ── Fix 3: The prompt preview is set via the esc() function which does a textContent
# equivalent. buildPrompt() returns a string that goes into clipboard — no template literal.
# The preview body is set via .innerHTML = esc(snippet) which safely escapes. That's OK.

# ── Fix 4: After the innerHTML assignment for panel, add textContent setter
# Find the end of the renderPanel function's innerHTML assignment
# The panel HTML ends with:   `;\n  const rmEl = ...
# We need to find the closing backtick of the template literal in renderPanel

# Strategy: find "  document.getElementById(\"panel\").innerHTML = `"
# then find the matching closing backtick (the one followed by ";\n")
# Insert our textContent line after the semicolon

INJECT_MARKER = '// Set roadmap safely (avoids backtick-in-template-literal issue)'
if INJECT_MARKER in content:
    print("Fix 2: Already patched. Skipping.")
else:
    # Find renderPanel function
    rp_start = content.find('function renderPanel()')
    if rp_start == -1:
        print("Fix 2: renderPanel NOT found")
    else:
        # Find the innerHTML assignment
        ih_start = content.find('document.getElementById("panel").innerHTML = `', rp_start)
        if ih_start == -1:
            print("Fix 2: innerHTML assignment NOT found")
        else:
            # Find its closing `; — the next backtick followed by ; on its own line
            # Look for \n  `;\n (the template literal close)
            close = content.find('\n  `;\n', ih_start)
            if close == -1:
                # try alternate
                close = content.find('`;\n}', ih_start)
                if close != -1:
                    insert_pos = close + 3  # after `;\n
                    inject = (
                        '\n  // Set roadmap safely (avoids backtick-in-template-literal issue)\n'
                        '  const rmEl = document.getElementById("roadmap-body");\n'
                        '  if (rmEl) rmEl.textContent = t.roadmap || "(no roadmap found)";\n'
                    )
                    content = content[:insert_pos] + inject + content[insert_pos:]
                    print("Fix 2: injected via alternate pattern  OK")
                else:
                    print("Fix 2: closing backtick NOT found")
            else:
                insert_pos = close + len('\n  `;\n')
                inject = (
                    '  // Set roadmap safely (avoids backtick-in-template-literal issue)\n'
                    '  const rmEl = document.getElementById("roadmap-body");\n'
                    '  if (rmEl) rmEl.textContent = t.roadmap || "(no roadmap found)";\n'
                )
                content = content[:insert_pos] + inject + content[insert_pos:]
                print("Fix 2: injected after template literal close  OK")

with open('prompt_builder.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done. File size:", len(content), "bytes")

# Quick validation
if 'rmEl.textContent = t.roadmap' in content:
    print("Validation: textContent setter found  OK")
else:
    print("Validation: textContent setter MISSING - need manual check")
    # Show the renderPanel area
    rp = content.find('function renderPanel()')
    print(content[rp:rp+2000])
