import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="14.1 Code Generation | GenAI Roadmap",
    page_icon="💻",
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
    .code-badge {
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
st.title("💻 14.1 Code Generation — Language Specs, Framework Constraints & Robustness")
st.markdown(
    """
Welcome to **Lesson 14.1: Code Generation**.  
Modern LLMs are proficient code generators, but without rigorous prompt constraints, they produce **untyped**, **insecure**, or **version-mismatched code** that fails in enterprise CI/CD pipelines.

Prompt engineering for code generation requires precise **language and framework versioning**, **explicit input/output type contracts**, **comprehensive error-handling mandates**, and **rejection of deprecated APIs**.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 5 Pillars of Enterprise Code Prompts
# ==============================================================================
st.header("1. The 5 Pillars of Production Code Generation Prompts", divider="rainbow")

col_cd1, col_cd2 = st.columns(2)

with col_cd1:
    with st.container(border=True):
        st.subheader("❌ Naive Code Prompt")
        st.code(
            '''Write a Python function to download an image from a URL and save it to disk.''',
            language="text",
        )
        st.error("🚨 Common Failures:")
        st.markdown(
            """
- Uses obsolete or blocking libraries (`urllib.request` instead of `httpx` or `aiohttp`)
- No timeout specified (causes worker threads to hang indefinitely on stalled sockets)
- No type annotations (`def download(url, path):`)
- Zero error handling for 404, 403, or network DNS failures
- Vulnerable to Path Traversal / SSRF attacks
"""
        )

with col_cd2:
    with st.container(border=True):
        st.subheader("✅ Hardened Production Code Prompt")
        st.code(
            '''<language>Python 3.12+</language>
<framework>httpx (asyncio)</framework>
<signature>
async def download_image(url: str, dest_path: Path, timeout_sec: float = 10.0) -> bool:
</signature>
<constraints>
- Use httpx.AsyncClient with explicit timeout.
- Validate that Content-Type header starts with 'image/'.
- Handle httpx.HTTPStatusError, httpx.TimeoutException, and OSError.
- Return False on error with logger.error trace; do not crash.
- Enforce strict typing (mypy compliant).
</constraints>''',
            language="xml",
        )
        st.success("🟢 Production Quality:")
        st.markdown(
            """
- Explicit runtime version (Python 3.12+)
- Non-blocking asynchronous I/O (`httpx`)
- Complete type signatures with `Path` and `bool`
- Comprehensive error handling and security headers
"""
        )

st.divider()

# ==============================================================================
# Section 2: Framework & Runtime Specification Matrix
# ==============================================================================
st.header("2. Framework & Runtime Specification Matrix", divider="rainbow")
st.markdown("LLMs are trained across decades of legacy code. You must pin versions to avoid deprecated paradigms:")

specs_df = pd.DataFrame({
    "Ecosystem": [
        "Python (Modern Async)",
        "TypeScript (Strict Frontend)",
        "SQL (PostgreSQL Analytics)",
        "Go (Microservices)",
        "Rust (Systems Programming)"
    ],
    "Mandatory Version Pinning": [
        "Python 3.11+, Pydantic v2 (not v1), SQLAlchemy 2.0 (not 1.4)",
        "TypeScript 5.0+, React 19 / Next.js 15 (Server Components)",
        "PostgreSQL 16+, ANSI SQL, CTEs (WITH queries)",
        "Go 1.22+, context.Context propagation, slog logging",
        "Rust 2021 Edition, Tokio 1.0, anyhow / thiserror"
    ],
    "Explicit Boundary Constraints": [
        "Use `model_validate()` not `parse_obj()`; use async/await",
        "No `any` types; strictly type React props and server actions",
        "Use parameterized bind parameters (`$1`, `$2`); no string formatting",
        "Never ignore `err`; check all errors immediately",
        "Use safe Rust only; forbid `unsafe {}` blocks"
    ]
})
st.dataframe(specs_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: Prompt-to-Verified-Code Pipeline", divider="rainbow")
st.markdown("How enterprise code prompts pass through automated linting and execution barriers:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        spec [label="Spec Prompt\n(Types + Framework + Pin)", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        llm [label="LLM Code Generation\n(Code-Specialized Model)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        ast_check [label="AST Syntax Parser\n(python -m py_compile)", fillcolor="#EAB308", fontcolor="#000000"]
        linter [label="Static Linter / Type Check\n(ruff / mypy)", fillcolor="#3B82F6", fontcolor="#FFFFFF"]
        sandbox [label="Ephemeral Test Sandbox\n(pytest execution)", fillcolor="#10B981", fontcolor="#FFFFFF"]

        spec -> llm -> ast_check -> linter -> sandbox
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Code Prompt Engineering Workbench
# ==============================================================================
st.header("4. Interactive Code Prompt Engineering Workbench", divider="rainbow")
st.markdown("Configure enterprise specifications and inspect the generated production-grade prompt and code:")

lang_choice = st.selectbox("Select Target Language & Runtime", ["Python 3.12 (Pydantic v2)", "TypeScript 5.3 (Node.js)", "PostgreSQL 16 (Relational)"])
error_handling = st.selectbox("Error Handling Philosophy", ["Defensive Result Object (Try/Except + Bool)", "Custom Exceptions with Logging", "Fail-Fast Assertions"])
include_docstrings = st.checkbox("Mandate Google-style Docstrings & Type Hints", value=True)

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("📄 Assembled Code Generation Prompt")
    if "Python" in lang_choice:
        code_prompt = f"""<system>
You are a Principal Software Engineer. Write clean, production-grade code.
</system>

<task>
Implement an API client function to fetch authenticated user metadata from a REST endpoint.
</task>

<specifications>
- Language: Python 3.12+
- Library: httpx (async) and pydantic v2
- Error Handling: {error_handling}
- Documentation: {'Google-style docstrings with parameter types' if include_docstrings else 'Minimal inline comments'}
- Quality: 100% type-hinted; strict mypy compliance.
</specifications>"""
    elif "TypeScript" in lang_choice:
        code_prompt = f"""<system>
You are a Staff Frontend Engineer. Output strict TypeScript code.
</system>

<task>
Implement a resilient fetch wrapper with exponential backoff retry logic.
</task>

<specifications>
- Language: TypeScript 5.3+
- Strict Mode: No 'any' types; use unknown with type guards
- Error Handling: {error_handling}
- Quality: Return Promise<Result<T, Error>>
</specifications>"""
    else:
        code_prompt = f"""<system>
You are a Principal Database Architect. Output optimized PostgreSQL DDL/DML.
</system>

<task>
Write a high-performance query to find top 5 customers by quarterly spending with month-over-month growth.
</task>

<specifications>
- Dialect: PostgreSQL 16+
- Optimization: Use Window functions and CTEs; include index recommendations
- Safety: Fully parameterized query
</specifications>"""

    st.code(code_prompt, language="xml")

with col_wb2:
    st.subheader("💻 Simulated High-Fidelity Code Output")
    if "Python" in lang_choice:
        sim_code = '''import httpx
from pydantic import BaseModel, EmailStr
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class UserMetadata(BaseModel):
    user_id: str
    email: EmailStr
    tier: str
    is_active: bool

async def fetch_user_metadata(
    client: httpx.AsyncClient, 
    base_url: str, 
    user_id: str, 
    auth_token: str
) -> Optional[UserMetadata]:
    """Fetches and validates user profile metadata from external auth microservice.

    Args:
        client: Shared asynchronous HTTP client instance.
        base_url: Microservice base domain URL.
        user_id: Unique customer identifier.
        auth_token: Bearer JWT credentials.

    Returns:
        UserMetadata object if successful; None if request fails or times out.
    """
    headers = {"Authorization": f"Bearer {auth_token}", "Accept": "application/json"}
    url = f"{base_url.rstrip('/')}/v1/users/{user_id}"
    
    try:
        response = await client.get(url, headers=headers, timeout=5.0)
        response.raise_for_status()
        return UserMetadata.model_validate(response.json())
    except (httpx.HTTPStatusError, httpx.TimeoutException, ValueError) as exc:
        logger.error(f"Failed to fetch metadata for user {user_id}: {exc}")
        return None'''
        st.code(sim_code, language="python")
    elif "TypeScript" in lang_choice:
        sim_code = '''type Result<T, E = Error> = 
  | { success: true; data: T } 
  | { success: false; error: E };

export async function fetchWithRetry<T>(
  url: string,
  options: RequestInit = {},
  maxRetries: number = 3
): Promise<Result<T, Error>> {
  let attempt = 0;
  while (attempt < maxRetries) {
    try {
      const response = await fetch(url, options);
      if (!response.ok) {
        throw new Error(`HTTP Error: ${response.status}`);
      }
      const data = (await response.json()) as T;
      return { success: true, data };
    } catch (err) {
      attempt++;
      if (attempt >= maxRetries) {
        return { success: false, error: err instanceof Error ? err : new Error(String(err)) };
      }
      const delay = Math.pow(2, attempt) * 200;
      await new Promise((res) => setTimeout(res, delay));
    }
  }
  return { success: false, error: new Error("Max retries exceeded") };
}'''
        st.code(sim_code, language="typescript")
    else:
        sim_code = '''-- Top 5 Customers by Quarterly Spend with Window Functions
WITH quarterly_spending AS (
    SELECT 
        customer_id,
        DATE_TRUNC('quarter', transaction_date) AS order_quarter,
        SUM(amount) AS total_spend,
        LAG(SUM(amount)) OVER (
            PARTITION BY customer_id 
            ORDER BY DATE_TRUNC('quarter', transaction_date)
        ) AS previous_quarter_spend
    FROM orders
    WHERE transaction_date >= NOW() - INTERVAL '1 year'
    GROUP BY customer_id, DATE_TRUNC('quarter', transaction_date)
)
SELECT 
    customer_id,
    order_quarter,
    total_spend,
    ROUND(((total_spend - previous_quarter_spend) / NULLIF(previous_quarter_spend, 0)) * 100, 2) AS mom_growth_pct
FROM quarterly_spending
ORDER BY total_spend DESC
LIMIT 5;

-- Recommended Covering Index:
-- CREATE INDEX idx_orders_cust_date_amount ON orders (transaction_date, customer_id) INCLUDE (amount);'''
        st.code(sim_code, language="sql")

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Prompt Generator for Code", divider="rainbow")
st.markdown("Python function that formats comprehensive coding prompts with architecture constraints:")

st.code(
    '''from typing import List

def generate_production_code_prompt(
    language: str,
    framework: str,
    function_signature: str,
    requirements: List[str],
    forbidden_patterns: List[str]
) -> str:
    """
    Assembles a robust prompt that forces the LLM to output secure, production-grade code.
    """
    req_bullets = "\\n".join(f"- {r}" for r in requirements)
    forbid_bullets = "\\n".join(f"- {f}" for f in forbidden_patterns)
    
    return f"""<system>
You are a Principal Software Architect.
Produce only production-ready {language} code. No conversational preambles.
</system>

<specifications>
- Language & Runtime: {language}
- Target Framework: {framework}
- Target Signature:
{function_signature}
</specifications>

<functional_requirements>
{req_bullets}
</functional_requirements>

<strict_constraints>
{forbid_bullets}
- Code must pass static type checking without errors or warnings.
- All external I/O must include timeouts and exception handling.
</strict_constraints>

<output_format>
Output the complete implementation inside a single fenced code block.
</output_format>"""

# Example usage:
prompt = generate_production_code_prompt(
    language="Python 3.12",
    framework="FastAPI + Pydantic v2",
    function_signature="@app.post('/items') async def create_item(payload: ItemCreate) -> ItemResponse:",
    requirements=["Validate item price > 0", "Persist to async session", "Return HTTP 201 Created"],
    forbidden_patterns=["Do not use blocking I/O", "Do not use Pydantic v1 methods (.dict(), .parse_obj())"]
)
print(prompt)
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
    with st.expander("Q1: Why do LLMs frequently hallucinate deprecated library methods, and how do you prevent it in prompts?", expanded=True):
        st.markdown(
            """
1. **Pre-training Training Bias**: An LLM's pre-training corpus contains millions of historical GitHub repositories and StackOverflow answers written years ago. If Pydantic v1 was used from 2018 to 2023, the model's weights contain more statistical probability for `.parse_obj()` than Pydantic v2's `.model_validate()`.
2. **Prompt Mitigation**: Explicitly specify the major version in `<framework>` (e.g. `Pydantic v2.6+`) and add an explicit negative constraint: *"Forbid deprecated methods like .dict() or .parse_obj(); use .model_dump() and .model_validate() exclusively."*
"""
        )
    with st.expander("Q2: How should unit tests be incorporated into code generation prompts?"):
        st.markdown(
            """
Instruct the model to generate both the implementation and a comprehensive **pytest / test suite** (including happy path, boundary values, and negative exception cases) in a separate test block. In automated agentic workflows, you can execute the generated tests inside an ephemeral Docker sandbox to verify that the code compiles and passes before merging.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we ask an LLM to generate code without specifying target framework versions?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, GPT-4 is smart enough to use the latest version automatically."*
- **Winning Response**: *"No. Major libraries frequently introduce breaking API changes across versions (e.g., SQLAlchemy 1.4 vs 2.0, LangChain 0.0 vs 0.3, Next.js Pages vs App Router). Without explicit version pinning in the prompt, the model will blend paradigms from older and newer documentation, producing code that fails to compile in modern development environments."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Framework Pinning**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Explicitly stating exact library major versions (e.g. Pydantic v2, React 19) in prompts to prevent the model from emitting legacy syntax.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Pre-training Version Bias**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The statistical tendency of LLMs to generate deprecated code because older paradigms appear more frequently in historic training data.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Defensive Code Prompting**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Requiring explicit timeout parameters, type annotations, and specific exception blocks in generated code.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 14.1 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Pin Versions Explicitly**: Always specify Python 3.12+, Pydantic v2, or Next.js 15 to eliminate legacy hallucinations.
2. **Strict Signatures**: Provide function signatures with complete type annotations upfront.
3. **Mandate Defensive I/O**: Require timeouts, error handling, and security header checks.
4. **Enforce Static Typing**: Instruct the model to produce 100% mypy / strict TypeScript compliant code.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I understand why LLMs default to deprecated APIs and how to prevent it.", value=True, key="chk_14_1_1")
    st.checkbox("I know how to define explicit type signatures and framework constraints in prompts.", value=True, key="chk_14_1_2")
    st.checkbox("I can structure code generation prompts that require defensive exception handling.", value=True, key="chk_14_1_3")
    st.checkbox("I understand how automated test execution sandboxes validate generated code.", value=True, key="chk_14_1_4")
