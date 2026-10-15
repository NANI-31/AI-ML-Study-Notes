import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="14.3 Advanced Coding Prompts | GenAI Roadmap",
    page_icon="🏢",
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
    .repo-pill {
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
st.title("🏢 14.3 Advanced Coding Prompts — Repository-Level Tasks & Migrations")
st.markdown(
    """
Welcome to **Lesson 14.3: Advanced Coding Prompts**.  
Real-world software engineering rarely operates on isolated 20-line functions. Production engineering involves **cross-file dependencies**, **project directory trees**, **database schema migrations**, and **large-scale framework upgrades**.

This lesson covers prompt engineering for repository-scale operations: multi-file context packing, automated API integration wrappers, schema migration scripts, and framework modernization.
"""
)

st.divider()

# ==============================================================================
# Section 1: The Challenges of Repository-Level Prompting
# ==============================================================================
st.header("1. Beyond Single Functions: Repository-Level Architecture", divider="rainbow")

col_rp1, col_rp2 = st.columns(2)

with col_rp1:
    with st.container(border=True):
        st.subheader("🧩 Multi-File Dependency Challenges")
        st.markdown(
            """
- **Import Chains**: Changing a type definition in `models/user.py` cascades across `services/auth.py`, `routers/api.py`, and `tests/test_auth.py`.
- **Circular Imports**: Naive multi-file code generation by LLMs frequently introduces circular module dependencies.
- **Context Window Limits**: You cannot paste an entire 100,000-line codebase into a single prompt. Context must be selectively pruned.
- **Architectural Conventions**: Code must adhere to existing project standards (naming conventions, dependency injection patterns, logging standards).
"""
        )

with col_rp2:
    with st.container(border=True):
        st.subheader("🎯 Enterprise Solutions & Prompt Patterns")
        st.markdown(
            """
1. **File Path Anchoring**: Explicitly delimit files with paths (`<file path="src/models.py">`).
2. **Skeleton & Interface Summaries**: Provide full AST signatures of adjacent files while omitting internal function bodies to save tokens.
3. **Targeted Diff Generation**: Require unified diffs (`diff --git a/...`) to avoid rewriting huge files.
4. **Step-by-Step Dependency Ordering**: Prompt the model to update base domain types *first*, then service layers, and test suites *last*.
"""
        )

st.divider()

# ==============================================================================
# Section 2: Advanced Coding Archetypes Matrix
# ==============================================================================
st.header("2. Advanced Coding Archetypes Matrix", divider="rainbow")
st.markdown("Specialized prompt structures for complex, multi-component engineering workflows:")

adv_archetypes_df = pd.DataFrame({
    "Task Archetype": [
        "1. Repository-Scale Refactoring",
        "2. Complex API Integration SDK",
        "3. Database Migration (Alembic / Prisma)",
        "4. Framework Modernization (e.g. Flask -> FastAPI)",
        "5. Automated Documentation & OpenAPI Generation"
    ],
    "Input Context Required": [
        "File tree directory listing + AST type interfaces of affected modules",
        "OpenAPI 3.1 JSON / YAML spec file + desired client ergonomics specification",
        "Current ORM model classes + target schema changes + existing migration state",
        "Legacy controller code + modern framework idiomatic standards rubric",
        "Source code classes, methods, docstrings, and response model types"
    ],
    "Mandatory Output Constraints": [
        "Multi-file diff patches specifying exact file paths and line offsets",
        "Async client class with type hints, auth interceptors, and retry middleware",
        "Reversible migration script containing both `upgrade()` and `downgrade()` steps",
        "Zero legacy paradigms; use async/await, dependency injection, and Pydantic models",
        "Complete Markdown or OpenAPI spec without placeholder comments"
    ]
})
st.dataframe(adv_archetypes_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: Repository Context Packing Pipeline", divider="rainbow")
st.markdown("How intelligent coding agents pack relevant repository context into prompts:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        repo [label="Entire Codebase\n(500+ Files)", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        indexer [label="AST & Dependency\nCall Graph Indexer", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        packer [label="Context Packer\n(Tree + Key Files + Skeletons)", fillcolor="#EAB308", fontcolor="#000000"]
        prompt [label="Repository Prompt\nwith Multi-File Tags", fillcolor="#3B82F6", fontcolor="#FFFFFF"]
        llm [label="LLM Reasoner\n(Architectural Plan)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        patch [label="Multi-File Git Diff\nPatch Set", fillcolor="#10B981", fontcolor="#FFFFFF"]

        repo -> indexer -> packer -> prompt -> llm -> patch
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Framework Migration Simulator
# ==============================================================================
st.header("4. Interactive Framework Migration Simulator", divider="rainbow")
st.markdown("Observe how an advanced migration prompt refactors a legacy synchronous Flask route into modern asynchronous FastAPI:")

col_m1, col_m2 = st.columns(2)

with col_m1:
    with st.container(border=True):
        st.subheader("❌ Legacy Synchronous Code (Flask)")
        st.code(
            '''from flask import Flask, request, jsonify
import time

app = Flask(__name__)

# Legacy route with blocking I/O and manual validation
@app.route('/orders', methods=['POST'])
def create_order():
    data = request.get_json()
    if not data or 'item_id' not in data or 'quantity' not in data:
        return jsonify({'error': 'Invalid payload'}), 400
        
    time.sleep(0.5)  # Blocking database write simulation
    order_id = "ORD-991"
    return jsonify({'order_id': order_id, 'status': 'created'}), 201''',
            language="python",
        )
        st.markdown("**Defects:** Blocking thread sleep, manual dictionary parsing, zero type safety, implicit error codes.")

with col_m2:
    with st.container(border=True):
        st.subheader("✅ Modern Asynchronous Code (FastAPI + Pydantic)")
        st.code(
            '''from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import asyncio

app = FastAPI(title="Order Microservice", version="2.0.0")

class OrderCreateRequest(BaseModel):
    item_id: str = Field(min_length=1, description="Inventory SKU")
    quantity: int = Field(ge=1, description="Number of items to order")

class OrderResponse(BaseModel):
    order_id: str
    status: str

@app.post("/orders", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def create_order(payload: OrderCreateRequest):
    # Non-blocking async I/O
    await asyncio.sleep(0.5)
    return OrderResponse(order_id="ORD-991", status="created")''',
            language="python",
        )
        st.markdown("**Modern Standards:** Non-blocking async I/O, declarative Pydantic v2 schemas, automated OpenAPI docs.")

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Repository Context Packer", divider="rainbow")
st.markdown("Python utility to format multiple repository files into structured XML for LLM ingestion:")

st.code(
    '''from typing import List, Dict
from pathlib import Path

def pack_repository_files_for_prompt(file_paths: List[str]) -> str:
    """
    Reads multiple source files and packages them into clean XML structures
    with unambiguous file path headers for multi-file LLM reasoning.
    """
    packed_blocks = ["<codebase_context>"]
    
    for path_str in file_paths:
        p = Path(path_str)
        if p.exists() and p.is_file():
            content = p.read_text(encoding="utf-8")
            packed_blocks.append(f'  <file path="{path_str}">')
            packed_blocks.append(f"{content}")
            packed_blocks.append("  </file>")
            
    packed_blocks.append("</codebase_context>")
    return "\\n".join(packed_blocks)

# Example demonstration of multi-file prompt assembly
files = ["src/config.py", "src/models/user.py", "src/services/auth.py"]
# In production, pass existing file paths to pack_repository_files_for_prompt(files)
print("Context packer successfully generates <codebase_context> blocks with strict <file path> tags.")
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
    with st.expander("Q1: How do coding agents (like Devin, Cursor, or Aider) handle codebases with 500,000 lines of code?", expanded=True):
        st.markdown(
            """
1. **Repository Map / AST Skeletons**: They generate an abstracted 'repo map' using tree-sitter, containing class definitions, method signatures, and export lists without implementation bodies. This fits the entire codebase overview into ~5,000 tokens.
2. **Selective File Retrieval (BM25 + Embeddings)**: When the user asks for a feature, semantic search and keyword indexers retrieve only the 3-5 relevant files and their immediate dependencies.
3. **Multi-Turn Editing / Diff Application**: The model generates git patch diffs rather than entire files, and the agent applies them deterministically using `patch` or git APIs.
"""
        )
    with st.expander("Q2: Why are Alembic database migrations particularly prone to LLM errors?"):
        st.markdown(
            """
LLMs frequently forget that relational databases require **backward-compatible migrations**:
- Adding a `NOT NULL` column without a `server_default` will instantly lock and crash existing production tables during deploy.
- LLMs often write the `upgrade()` migration step but leave `downgrade()` empty (`pass`), preventing rollback operations during failed deployments.
- **Prompt Defense**: Mandate reversible migrations with explicit `server_default` values for all new non-nullable columns.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we ask the LLM to rewrite all 15 files in our project at once?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, give it the entire directory and ask it to modernize everything in one turn."*
- **Winning Response**: *"Never. Asking for massive multi-file rewrites in a single prompt causes token truncation, forgotten files, and hallucinated import paths. Production repo migrations must be broken into a phased DAG: Phase 1 updates shared schemas, Phase 2 updates internal business logic, and Phase 3 updates API routes and tests, with automated linting and test passes executed between each phase."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Repo Map / Tree-Sitter**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** An abstracted overview of a codebase containing only symbols, class signatures, and imports to save tokens.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Unified Diff Patch**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Standard git diff format showing additions (+) and deletions (-) to modify files without regenerating thousands of unchanged lines.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Phased Migration DAG**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Decomposing large repo migrations into sequential stages (Types -> Logic -> APIs -> Tests) with compiler checks between steps.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 14.3 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Repository Context**: Multi-file engineering requires file path tags, repo maps, and AST skeletons.
2. **Unified Diffs**: Mandate diffs rather than full file rewrites to prevent truncation and save tokens.
3. **Reversible Migrations**: Ensure database migrations include valid `downgrade()` methods and defaults.
4. **Phased Execution**: Migrate complex architectures stage-by-stage using automated test verification gates.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I know how to pack multiple source files into structured XML prompts.", value=True, key="chk_14_3_1")
    st.checkbox("I understand why coding agents use AST repo maps rather than full text files.", value=True, key="chk_14_3_2")
    st.checkbox("I can structure prompts for reversible database migrations.", value=True, key="chk_14_3_3")
    st.checkbox("I understand the phased DAG approach to multi-file framework modernization.", value=True, key="chk_14_3_4")
