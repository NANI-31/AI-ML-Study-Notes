import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="14.2 Code Analysis | GenAI Roadmap",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        will-change: transform;
        contain: layout style;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .analysis-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.title("🔍 14.2 Code Analysis — Debugging, Refactoring & Test Generation")
st.markdown(
    """
Welcome to **Lesson 14.2: Code Analysis**.  
Software engineering is far more than writing new code from scratch—over 70% of an engineer's time is spent **reading**, **debugging**, **reviewing**, and **refactoring** existing legacy systems.

In this lesson, we master prompt engineering for the **6 Core Code Analysis Tasks**: line-by-line explanation, root-cause debugging, algorithmic optimization, architectural refactoring, security code review, and automated test suite synthesis.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 6 Pillars of Code Analysis Prompting
# ==============================================================================
st.header("1. The 6 Pillars of Code Analysis Prompting", divider="rainbow")
st.markdown("Structuring analytical prompts for specific software engineering workflows:")

analysis_tasks_df = pd.DataFrame({
    "Analysis Mode": [
        "1. Code Explanation",
        "2. Root-Cause Debugging",
        "3. Architectural Refactoring",
        "4. Performance Optimization",
        "5. Security Review (SAST)",
        "6. Automated Test Generation"
    ],
    "Key Prompt Objective": [
        "Unpack complex algorithms for juniors, decode obscure legacy code, document intent",
        "Trace stack traces, find off-by-one errors, identify race conditions, diagnose NoneType crashes",
        "Decouple monoliths, apply SOLID principles, introduce design patterns, improve readability",
        "Reduce Big-O time and space complexity, eliminate N+1 queries, vectorize loops",
        "Detect OWASP vulnerabilities, audit sanitization, identify memory leaks and buffer overflows",
        "Generate exhaustive unit tests covering edge cases, happy paths, and parameterized fixtures"
    ],
    "Prompt Engineering Invariants": [
        "Require analogies and step-by-step logic walkthrough; avoid repeating code verbatim",
        "Force the model to state the exact line number, root cause, and minimal reproducible fix",
        "Provide unified git diff output; preserve public API backward compatibility",
        "Require Big-O before vs after comparison; mandate memory profile estimations",
        "Classify findings by Common Weakness Enumeration (CWE) and CVSS severity ratings",
        "Use modern test runners (pytest, vitest); mandate mock fixtures for external I/O"
    ]
})
st.dataframe(analysis_tasks_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Production Prompt Blueprints for Debugging & Testing
# ==============================================================================
st.header("2. Production Prompt Blueprints: Debugging & Test Synthesis", divider="rainbow")

col_bp1, col_bp2 = st.columns(2)

with col_bp1:
    with st.container(border=True):
        st.subheader("🐛 Systematic Debugging Prompt")
        st.code(
            '''<system>
You are a Principal Software Debugging Specialist.
</system>

<instructions>
Diagnose the bug in the code snippet provided in <code> given the error in <error_trace>.
Follow this exact 3-step diagnostic rubric:
1. ROOT CAUSE: Identify the exact line number and variable causing the fault.
2. FAILURE MECHANISM: Explain why the state corrupted during runtime.
3. SURGICAL FIX: Provide the corrected code with minimal lines changed.
</instructions>

<code>
def calculate_user_discount(user_profile, cart_total):
    if user_profile['tier'] == 'VIP' or cart_total > 100:
        return cart_total * 0.8
    elif user_profile['tier'] == 'MEMBER':
        return cart_total * 0.9
    return cart_total
</code>

<error_trace>
KeyError: 'tier' at line 2 in calculate_user_discount
user_profile = {"name": "Guest", "is_anonymous": True}
cart_total = 150.00
</error_trace>''',
            language="xml",
        )

with col_bp2:
    with st.container(border=True):
        st.subheader("🧪 Exhaustive Unit Test Synthesis Prompt")
        st.code(
            '''<system>
You are a Staff QA Engineer. Write pytest unit test suites.
</system>

<instructions>
Generate a comprehensive pytest suite for the function inside <code>.
Requirements:
- 100% branch and edge-case coverage.
- Include happy path, boundary values, empty payloads, and exception triggers.
- Use @pytest.mark.parametrize for multiple input matrices.
- Use pytest-mock (mocker fixture) for external database and network calls.
- Output ONLY executable Python code.
</instructions>

<code>
async def transfer_funds(ledger_db, source_acct: str, dest_acct: str, amount_usd: float) -> str:
    if amount_usd <= 0:
        raise ValueError("Transfer amount must be positive")
    return await ledger_db.atomic_transfer(source_acct, dest_acct, amount_usd)
</code>''',
            language="xml",
        )

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: Code Analysis to Automated Verification Pipeline", divider="rainbow")
st.markdown("How code analysis prompts integrate into developer tooling and CI/CD pipelines:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        source [label="Target Codebase\n& Stack Trace", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        prompt [label="Analysis Prompt\n(Review / Test / Debug)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        llm [label="LLM Code Reasoner\n(Claude 3.5 / GPT-4o)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        patch [label="Unified Diff Patch\nor Pytest Suite", fillcolor="#EAB308", fontcolor="#000000"]
        ci [label="CI Test Runner\n(pytest -v --cov)", fillcolor="#3B82F6", fontcolor="#FFFFFF"]
        verdict [label="Merge Ready\nor Auto-Feedback Loop", fillcolor="#10B981", fontcolor="#FFFFFF"]

        source -> prompt -> llm -> patch -> ci -> verdict
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Code Analysis Workbench
# ==============================================================================
st.header("4. Interactive Code Analysis Workbench", divider="rainbow")
st.markdown("Select an analysis task to see how structured prompts extract actionable engineering insights:")

analysis_mode = st.selectbox(
    "Choose Analysis Objective",
    [
        "Bug Diagnosis: Unhandled KeyError in Dictionary",
        "Performance Optimization: N+1 Database Query in ORM",
        "Security SAST Review: SQL Injection Vulnerability",
        "Automated Pytest Generation: Bank Transfer Function"
    ]
)

col_aw1, col_aw2 = st.columns([1, 1])

with col_aw1:
    st.subheader("📥 Target Code Snippet")
    if "KeyError" in analysis_mode:
        sample_code = """def get_tax_rate(shipping_address):
    # Bug: Raises KeyError if 'state' is omitted (e.g. international orders)
    state = shipping_address['state']
    return TAX_TABLE[state]"""
    elif "N+1" in analysis_mode:
        sample_code = """# Performance Issue: N+1 query loop
orders = db.query(Order).filter(Order.status == 'PENDING').all()
for order in orders:
    customer_name = order.customer.name  # Triggers 1 new query per order!
    send_notification(customer_name, order.id)"""
    elif "SQL Injection" in analysis_mode:
        sample_code = """def search_products(query_term):
    # Security: Raw string formatting directly into SQL string
    query = f"SELECT * FROM items WHERE name LIKE '%{query_term}%' AND is_active = TRUE"
    return db.execute(query).fetchall()"""
    else:
        sample_code = """def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise ValueError("Withdrawal amount must be positive")
    if amount > balance:
        raise InsufficientFundsError("Cannot withdraw more than balance")
    return balance - amount"""

    st.code(sample_code, language="python")

with col_aw2:
    st.subheader("📤 LLM Analytical Output")
    if "KeyError" in analysis_mode:
        st.markdown(
            """
**Root Cause Analysis:**
- **File / Line**: Line 3: Direct dictionary indexing `shipping_address['state']`.
- **Fault Mechanism**: Assumes domestic US addresses where `state` key is guaranteed. International or guest checkout payloads omitting `state` trigger an unhandled `KeyError`.
- **Surgical Refactoring**:
```python
def get_tax_rate(shipping_address: dict) -> float:
    # Use .get() with fallback or handle None
    state = shipping_address.get('state')
    if not state or state not in TAX_TABLE:
        return DEFAULT_INTERNATIONAL_TAX_RATE
    return TAX_TABLE[state]
```
"""
        )
    elif "N+1" in analysis_mode:
        st.markdown(
            """
**Performance Diagnostic:**
- **Issue**: Classic $N+1$ query anti-pattern. If there are 1,000 pending orders, the application fires $1 + 1,000 = 1,001$ individual SQL round-trips.
- **Complexity**: Time scales linearly $O(N)$ with database network latency.
- **Optimized SQLAlchemy Query (Eager Loading)**:
```python
# Eager-load customer relationship using joinedload (1 single SQL query)
from sqlalchemy.orm import joinedload

orders = (
    db.query(Order)
    .options(joinedload(Order.customer))
    .filter(Order.status == 'PENDING')
    .all()
)
for order in orders:
    send_notification(order.customer.name, order.id)
```
"""
        )
    elif "SQL Injection" in analysis_mode:
        st.markdown(
            """
**SAST Security Audit:**
- **CWE Identifier**: **CWE-89**: Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection').
- **Severity**: **CRITICAL (CVSS 9.8)**.
- **Exploitation**: Passing `'; DROP TABLE items; --` in `query_term` executes arbitrary SQL commands.
- **Remediated Parameterized Implementation**:
```python
from sqlalchemy import text

def search_products(query_term: str):
    # Parameterized query with bind parameters
    stmt = text("SELECT * FROM items WHERE name ILIKE :pattern AND is_active = TRUE")
    return db.execute(stmt, {"pattern": f"%{query_term}%"}).fetchall()
```
"""
        )
    else:
        st.markdown(
            """
**Generated Pytest Suite:**
```python
import pytest

@pytest.mark.parametrize("balance,amount,expected", [
    (100.0, 50.0, 50.0),    # Standard partial withdrawal
    (50.0, 50.0, 0.0),      # Complete balance withdrawal
    (1000.0, 0.01, 999.99), # Minimal amount
])
def test_withdraw_success(balance, amount, expected):
    assert withdraw(balance, amount) == pytest.approx(expected)

def test_withdraw_negative_amount_raises_value_error():
    with pytest.raises(ValueError, match="must be positive"):
        withdraw(100.0, -10.0)

def test_withdraw_zero_amount_raises_value_error():
    with pytest.raises(ValueError, match="must be positive"):
        withdraw(100.0, 0.0)

def test_withdraw_insufficient_funds_raises_error():
    with pytest.raises(InsufficientFundsError):
        withdraw(50.0, 100.0)
```
"""
        )

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Automated PR Review Agent", divider="rainbow")
st.markdown("Python script orchestrating automated code audits on git diffs:")

st.code(
    '''import json
from typing import Dict, Any

CODE_REVIEW_SYSTEM_PROMPT = """You are a Principal Security and Performance Code Reviewer.
Analyze the provided git diff inside <git_diff>.
Output a single valid JSON object matching this schema:
{
  "approved": bool,
  "summary": str,
  "issues_found": [
    {
      "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
      "category": "SECURITY" | "PERFORMANCE" | "CORRECTNESS" | "STYLE",
      "line_hint": str,
      "description": str,
      "recommended_fix": str
    }
  ]
}
Return ONLY valid JSON.
"""

def build_pr_review_payload(git_diff_text: str) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": CODE_REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": f"<git_diff>\\n{git_diff_text}\\n</git_diff>"}
        ],
        "temperature": 0.1
    }

# Example git diff for review
sample_diff = """
--- a/auth.py
+++ b/auth.py
@@ -12,4 +12,4 @@
 def check_password(input_pwd, stored_hash):
-    return bcrypt.checkpw(input_pwd.encode(), stored_hash.encode())
+    return input_pwd == stored_hash  # Temporarily disabled bcrypt for speed
"""
print(build_pr_review_payload(sample_diff)["messages"][1]["content"])
''',
    language="python",
)

st.divider()

# ==============================================================================
# Section 6: Technical Interview Preparation
# ==============================================================================
st.header("6. Technical Interview Preparation", divider="rainbow")

tab_q, tab_t, tab_f = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with tab_q:
    with st.expander("Q1: Why is asking an LLM to generate unit tests more reliable than asking it to verify code correctness directly?", expanded=True):
        st.markdown(
            """
1. **Execution Grounding**: LLMs can hallucinate that buggy code is correct because the code "looks plausible" to the language model. When an LLM generates a **concrete test suite**, those tests can be executed by a deterministic Python interpreter (`pytest`).
2. **Deterministic Truth**: The test suite either passes or fails in runtime reality. If a test fails, the error message and trace can be re-fed to the LLM to fix the bug, grounding the entire self-healing loop in empirical truth.
"""
        )
    with st.expander("Q2: How do you structure a prompt to prevent an LLM from doing unwanted 'drive-by' refactoring during bug fixes?"):
        st.markdown(
            """
Add an explicit **Scope Isolation Constraint**:
- *"Make the minimal changes necessary to fix the bug."*
- *"Do NOT rename variables, alter docstrings, reformat whitespace, or introduce new libraries unless strictly required to resolve the runtime exception."*
- *"Output a unified git diff showing only the targeted lines changed."*
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we rely on LLMs to catch subtle cryptographic timing attacks or race conditions without specialized prompts?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, frontier models spot all vulnerabilities automatically."*
- **Winning Response**: *"No. LLMs naturally focus on common surface-level patterns (like SQL injection or raw passwords). Subtle concurrency bugs (race conditions, deadlock potential) or cryptographic timing attacks (using standard `==` instead of `hmac.compare_digest()`) are frequently overlooked unless the prompt explicitly primes the model with targeted threat-modeling checklists (e.g., 'Specifically audit for thread safety and timing attacks')."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Test-Driven LLM Repair**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The engineering workflow of having the model write tests first, running them in a real interpreter, and fixing code based on test failures.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Drive-By Refactoring**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The common LLM failure where it rewrites surrounding working code, renames functions, and breaks public APIs while fixing a minor bug.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**CWE Tagging**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Mandating that LLM security reviewers classify vulnerabilities using standard Common Weakness Enumeration identifiers (e.g. CWE-89).")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 14.2 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Diverse Analysis Workflows**: Code analysis spans debugging, optimization, SAST review, and test synthesis.
2. **Minimal Surgical Patches**: Constrain models to output minimal unified diffs rather than full file rewrites.
3. **Ground in Test Execution**: Pair LLM test generation with real-world test runners (e.g. `pytest`).
4. **Threat Modeling Guidance**: Specifically instruct security prompts to check for subtle concurrency and crypto issues.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I know how to construct systematic 3-step root-cause debugging prompts.", value=True, key="chk_14_2_1")
    st.checkbox("I can structure prompts to generate comprehensive pytest suites with fixtures and edge cases.", value=True, key="chk_14_2_2")
    st.checkbox("I understand how to prevent unwanted drive-by refactoring in bug fixes.", value=True, key="chk_14_2_3")
    st.checkbox("I can implement an automated PR review agent payload in Python.", value=True, key="chk_14_2_4")
