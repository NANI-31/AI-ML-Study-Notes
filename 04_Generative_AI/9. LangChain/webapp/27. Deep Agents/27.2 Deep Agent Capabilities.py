import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="27.2 Deep Agent Capabilities - LangChain Architecture",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for filesystem telemetry and task decomposition cards
st.markdown("""
<style>
    .task-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-task-done {
        background-color: #065F46;
        color: #34D399;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-task-active {
        background-color: #312E81;
        color: #C7D2FE;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class DeepCapabilitiesSimulator:
    """
    Simulation engine showcasing task decomposition, workspace filesystem operations,
    and progressive context compression during long-running tasks.
    """
    def decompose_task(self, task_name: str) -> List[Dict[str, Any]]:
        if "refactor" in task_name.lower():
            return [
                {"id": 1, "title": "Index Workspace AST", "type": "FS_READ", "expected_tokens": 12000, "compressed_tokens": 250},
                {"id": 2, "title": "Spawn Test Subagent", "type": "SUBAGENT", "expected_tokens": 25000, "compressed_tokens": 180},
                {"id": 3, "title": "Apply Unified Diff Patch", "type": "FS_WRITE", "expected_tokens": 8000, "compressed_tokens": 320},
                {"id": 4, "title": "Verify Regression Suite", "type": "TEST_EXEC", "expected_tokens": 15000, "compressed_tokens": 150}
            ]
        else:
            return [
                {"id": 1, "title": "Scan Requirements & Schema", "type": "FS_READ", "expected_tokens": 6000, "compressed_tokens": 200},
                {"id": 2, "title": "Synthesize Implementation", "type": "FS_WRITE", "expected_tokens": 18000, "compressed_tokens": 400},
                {"id": 3, "title": "Run Integration Verification", "type": "TEST_EXEC", "expected_tokens": 10000, "compressed_tokens": 160}
            ]

    def calculate_compression_savings(self, subtasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        raw_total = sum(t["expected_tokens"] for t in subtasks)
        compressed_total = sum(t["compressed_tokens"] for t in subtasks)
        return {
            "raw_tokens": raw_total,
            "compressed_tokens": compressed_total,
            "savings_pct": round((1 - (compressed_total / raw_total)) * 100, 1)
        }


# --- LESSON HEADER ---
st.title("27.2 Deep Agent Capabilities: Tooling & Execution Engine")
st.caption("Chapter 27: Deep Agents | Domain: 9. LangChain Orchestration & LCEL")
st.markdown("""
A Deep Agent is distinguished not merely by its cognitive intent, but by the tangible operational capabilities
embedded in its execution environment. To independently complete complex engineering mandates, a Deep Agent must possess
fine-grained task decomposition, subagent process management, safe filesystem read/write operations, aggressive context compression,
and crash-proof long-running durability.

In this lesson, you will master:
1. Structured **Planning** and dependency mapping for multi-phase tasks.
2. Recursive **Task Decomposition** into verifiable, atomic units of work.
3. Managing the lifecycle of **Subagent Spawning** and garbage collection.
4. Safe **Filesystem Operations**: AST indexing, line-level editing, diffing, and git operations.
5. Lossless and lossy **Context Compression** techniques to prevent memory exhaustion.
6. Architecting **Long-Running Tasks** with checkpoint persistence and pause/resume capability.
""")
st.divider()

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. End-to-End Architectural Blueprint: Deep Agent Operational Tooling & State Engine", divider="rainbow")
st.markdown("""
A production Deep Agent interfaces with a workspace environment through a structured capability layer:
reading files, spawning child processes, persisting state snapshots, and compressing context into long-term working memory.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    cap_dot = """
    digraph DeepCapabilitiesBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

        subgraph cluster_planning_phase {
            label="1. Planning & Decomposition Engine";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            goal [label="High-Level Goal", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            decomposer [label="Task Decomposer\\nSplits goal into Atomic Subtasks", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            task_dag [label="Subtask DAG\\n[T1: Read] -> [T2: Subagent] -> [T3: Patch]", fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
        }

        subgraph cluster_operational_tools {
            label="2. Workspace Operational Capabilities";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            fs_ops [label="Filesystem Tools\\nread_file, write_patch, git_diff", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            subagent_pool [label="Subagent Spawner\\nSpawns isolated child process", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            test_runner [label="Sandbox Test Runner\\nExecutes pytest / npm test", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
        }

        subgraph cluster_memory_lifecycle {
            label="3. Memory Compression & Durable Checkpoint";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            compressor_unit [label="Context Compressor\\nTruncates 50k tokens to 400-token artifact", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            checkpointer_db [label="Durable Checkpoint Storage\\nPostgres / Sqlite snapshot", fillcolor="#065F46", fontcolor="#D1FAE5", color="#34D399"];
        }

        goal -> decomposer [label="Ingress"];
        decomposer -> task_dag [label="Generates"];
        task_dag -> fs_ops [label="Execute T1"];
        task_dag -> subagent_pool [label="Execute T2"];
        fs_ops -> test_runner [label="Apply Changes"];
        test_runner -> compressor_unit [label="Raw Test Output"];
        compressor_unit -> checkpointer_db [label="Persist State"];
        checkpointer_db -> task_dag [label="Advance Next Task"];
    }
    """
    st.graphviz_chart(cap_dot, use_container_width=True)

# --- SECTION 2: PLANNING ---
st.header("2. Planning", divider="rainbow")
st.markdown("""
In a Deep Agent, **Planning** is the continuous process of aligning operational tool calls with strategic objectives.
A plan is not a static text string; it is a structured data object containing pre-conditions, estimated tokens,
active blockers, and verification criteria.
""")

col_pl1, col_pl2 = st.columns(2)
with col_pl1:
    st.markdown("""
    #### Core Planning Principles:
    - **Pre-Flight Validation**: Checking repository existence, installed dependencies, and API credentials before planning tasks.
    - **Explicit Verification Criteria**: Every planned task must define how success is verified (e.g., `'Zero linter errors'` or `'HTTP 200 returned'`).
    - **Progressive Elaboration**: Planning immediate next steps in granular detail, while keeping distant future milestones at a high level.
    """)

with col_pl2:
    with st.container(border=True):
        st.markdown("📋 **Pydantic Task Plan Schema**")
        st.code("""
from pydantic import BaseModel, Field
from typing import List, Optional

class AtomicTask(BaseModel):
    task_id: int
    description: str
    tool_required: str
    success_criterion: str
    is_completed: bool = False

class ExecutionPlan(BaseModel):
    objective: str
    tasks: List[AtomicTask]
    current_index: int = 0
        """, language="python")

# --- SECTION 3: TASK DECOMPOSITION ---
st.header("3. Task Decomposition", divider="rainbow")
st.markdown("""
Large software engineering tasks cannot be executed in a single LLM call. **Task Decomposition** breaks an open-ended
goal into atomic, independent units of execution.
""")

st.subheader("Example: Decomposing 'Add Stripe Webhook Handling'")
decomp_df = pd.DataFrame({
    "Step": ["1. Workspace Audit", "2. Signature Verifier", "3. Event Dispatcher", "4. Unit Tests", "5. Git Commit"],
    "Action": ["Locate existing router and auth middleware", "Write cryptographic signature validation helper", "Implement POST /webhooks/stripe route", "Write pytest mocking Stripe event payloads", "Format code, stage files, and generate commit"],
    "Tool Used": ["grep_search, read_file", "write_file", "write_file", "run_test_sandbox", "git_commit"],
    "Verification": ["Router file path confirmed", "Unit test passes with dummy secret", "Route registered in app", "4/4 tests pass", "Clean git status"]
})
st.dataframe(decomp_df, use_container_width=True)

# --- SECTION 4: SUBAGENT SPAWNING ---
st.header("4. Subagent Spawning", divider="rainbow")
st.markdown("""
A primary capability of Deep Agents is the ability to programmatically spawn child agents.
When an agent encounters a large, uncertain sub-problem (e.g., investigating why 12 integration tests are failing),
it delegates the entire debugging loop to a specialized subagent.
""")

sub_sp_dot = """
digraph SubagentLifecycle {
    rankdir=LR;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    parent [label="Primary Controller\\nNeeds test failure analysis", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
    spawn_gate [label="Subagent Factory\\nAllocates isolated RAM & sandbox", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
    child_agent [label="Child Subagent\\nRuns 15 debugger turns", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
    artifact_return [label="Return Artifact\\n'Bug caused by unescaped JSON in L42'", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];

    parent -> spawn_gate [label="spawn_subagent(task)"];
    spawn_gate -> child_agent [label="Executes in isolation"];
    child_agent -> artifact_return [label="Harvests finding"];
    artifact_return -> parent [label="Clean handover; destroy child"];
}
"""
st.graphviz_chart(sub_sp_dot, use_container_width=True)

# --- SECTION 5: FILESYSTEM OPERATIONS ---
st.header("5. Filesystem Operations", divider="rainbow")
st.markdown("""
Unlike toy agents that output whole-file code snippets for the user to copy-paste, Deep Agents directly operate
on repository filesystems using specialized file tools.
""")

col_fs1, col_fs2 = st.columns(2)
with col_fs1:
    st.markdown("""
    #### Production Filesystem Tool Suite:
    - **`read_file_chunk(path, start_line, end_line)`**: Reads specific line slices rather than dumping 5,000-line files into context.
    - **`grep_search(query, path_glob)`**: Fast regex ripgrep across repository trees to locate symbols and imports.
    - **`apply_unified_diff(path, diff_content)`**: Applies surgical line replacements rather than overwriting entire files, preventing accidental deletion of unrelated code.
    - **`git_status_and_diff()`**: Inspects workspace changes to verify safety before making commits.
    """)

with col_fs2:
    with st.container(border=True):
        st.markdown("📝 **Surgical Diff Patch Example**")
        st.code("""
--- a/src/auth.py
+++ b/src/auth.py
@@ -14,6 +14,8 @@
 def verify_token(token: str) -> bool:
+    if not token or len(token) < 16:
+        return False
     return jwt.decode(token, SECRET, algorithms=["HS256"])
        """, language="diff")

# --- SECTION 6: CONTEXT COMPRESSION ---
st.header("6. Context Compression", divider="rainbow")
st.markdown("""
A Deep Agent operating over 4 hours will generate over 200,000 tokens of file contents, grep results, test outputs,
and terminal diagnostics. **Context Compression** condenses this firehose into dense, informative representations.
""")

st.subheader("Techniques for Long-Running Context Compression")
comp_c1, comp_c2, comp_c3 = st.columns(3)
with comp_c1:
    with st.container(border=True):
        st.markdown("🎯 **Observation Summarization**")
        st.markdown("When a tool returns 1,000 lines of terminal output, an automated summarizer condenses it to: `'Test suite: 48 passed, 2 failed. Errors in test_auth.py:42.'`")

with comp_c2:
    with st.container(border=True):
        st.markdown("🗜️ **Lossless AST Extraction**")
        st.markdown("Instead of storing whole Python files, extract and retain only the function signatures, docstrings, and class hierarchies.")

with comp_c3:
    with st.container(border=True):
        st.markdown("🧹 **State Rolling Window**")
        st.markdown("Drop conversation turns older than 10 iterations while appending critical milestones to the structured system scratchpad.")

# --- SECTION 7: LONG-RUNNING TASKS ---
st.header("7. Long-Running Tasks", divider="rainbow")
st.markdown("""
Enterprise tasks can take hours or span days. A Deep Agent must possess durable persistence, surviving process
restarts, container redeployments, and network hiccups.
""")

col_lr1, col_lr2 = st.columns(2)
with col_lr1:
    st.markdown("""
    #### Durability Architecture:
    1. **Checkpointed State Graph**: Every turn is committed to an external database (Postgres, SQLite) via LangGraph checkpointers.
    2. **Resumable Threads**: If the server crashes at 02:00 AM, the agent re-hydrates its exact state upon restart and resumes milestone 4 without starting over.
    3. **Human Pause-and-Resume**: The agent can pause execution, send a Slack webhook requesting user confirmation for a database migration, and sleep until the approval callback arrives.
    """)

with col_lr2:
    with st.container(border=True):
        st.markdown("💾 **LangGraph Checkpointer Persistence**")
        st.code("""
from langgraph.checkpoint.sqlite import SqliteSaver

# Persist all execution turns to durable SQLite database
with SqliteSaver.from_conn_string("deep_agent_state.db") as memory:
    app = workflow.compile(checkpointer=memory)
    
    # Run with unique thread ID
    config = {"configurable": {"thread_id": "mission-7721"}}
    app.invoke(initial_state, config=config)
    
    # Later: Resume seamlessly using the same thread ID!
        """, language="python")

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Deep Agent Task Decomposer & Workspace Controller", divider="rainbow")
st.markdown("""
Below is a runnable Python implementation of a Deep Agent workspace controller demonstrating
task decomposition, simulated filesystem operations, and automated context compression.
""")

prod_cap_code = '''from typing import List, Dict, Any
from pydantic import BaseModel

class SubTask(BaseModel):
    id: int
    name: str
    action_type: str
    status: str = "PENDING"
    raw_output: str = ""
    compressed_summary: str = ""

class DeepWorkspaceController:
    def __init__(self, objective: str):
        self.objective = objective
        self.tasks: List[SubTask] = self._decompose_objective()
        self.compressed_scratchpad: List[str] = []

    def _decompose_objective(self) -> List[SubTask]:
        # Programmatic decomposition into atomic actions
        return [
            SubTask(id=1, name="Read src/models.py", action_type="FS_READ"),
            SubTask(id=2, name="Generate UserSchema Pydantic model", action_type="CODE_GEN"),
            SubTask(id=3, name="Execute unit test suite", action_type="TEST_RUN")
        ]

    def execute_task(self, task_id: int, simulated_raw_data: str) -> dict:
        task = next(t for t in self.tasks if t.id == task_id)
        task.raw_output = simulated_raw_data
        
        # Automatic Context Compression: Distill 500 lines to 1 concise line
        task.compressed_summary = f"Completed {task.name}: Verified without errors."
        task.status = "COMPLETED"
        
        # Append only the compressed summary to working memory
        self.compressed_scratchpad.append(task.compressed_summary)
        
        return {
            "task_id": task.id,
            "raw_token_count": len(simulated_raw_data.split()),
            "compressed_token_count": len(task.compressed_summary.split()),
            "summary": task.compressed_summary
        }

    def get_persisted_state(self) -> dict:
        return {
            "objective": self.objective,
            "tasks_remaining": sum(1 for t in self.tasks if t.status == 'PENDING'),
            "working_memory": self.compressed_scratchpad
        }
'''
st.code(prod_cap_code, language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Long-Running Task Decomposition & Context Compression Simulator", divider="rainbow")
st.markdown("""
Simulate how a Deep Agent decomposes complex software mandates into atomic subtasks,
and observe how active context compression slashes memory consumption across long-running sessions.
""")

cap_engine = DeepCapabilitiesSimulator()

col_cw1, col_cw2 = st.columns([1, 1])

with col_cw1:
    st.subheader("⚙️ Task Specification & Execution")
    preset_goals = [
        "Refactor Authentication Middleware to OAuth2",
        "Implement Stripe Subscription Webhook Integration",
        "Migrate SQLite Schema to Postgres with Unit Tests"
    ]
    chosen_goal = st.selectbox("Select High-Level Enterprise Goal:", preset_goals)
    decompose_btn = st.button("🔨 Decompose & Calculate Compression Savings", type="primary", use_container_width=True)

with col_cw2:
    st.subheader("📊 Operational Capabilities Telemetry")
    if decompose_btn:
        subtasks = cap_engine.decompose_task(chosen_goal)
        savings = cap_engine.calculate_compression_savings(subtasks)

        st.markdown(
            f"<div class='task-card'><span class='badge-task-active'>PLAN DECOMPOSED</span>"
            f"<br><br><b>Atomic Subtasks Generated:</b> {len(subtasks)}"
            f"<br><b>Context Compression Efficiency:</b> {savings['savings_pct']}% Token Reduction</div>",
            unsafe_allow_html=True
        )

        col_sav1, col_sav2 = st.columns(2)
        with col_sav1:
            st.metric("Raw Tool Return Tokens", f"{savings['raw_tokens']:,}")
        with col_sav2:
            st.metric("Compressed Working Memory", f"{savings['compressed_tokens']:,}", delta=f"-{savings['savings_pct']}%")

        st.markdown("##### 📋 Decomposed Task Execution Plan")
        task_table = pd.DataFrame(subtasks)
        st.dataframe(task_table[["id", "title", "type", "expected_tokens", "compressed_tokens"]], use_container_width=True)
    else:
        st.info("Select a goal and click 'Decompose & Calculate Compression Savings' to inspect the execution tree.")

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: Why should filesystem tools use line-based diffs rather than full-file writes?"):
    st.markdown("""
    **Answer**: Full-file writes require the LLM to output the entire file content, which wastes massive output token quotas,
    increases generation latency, and frequently introduces accidental regressions (e.g., dropping existing helper functions
    or imports). Line-based surgical diffs (like unified diffs or regex search-and-replace) only output the changed lines,
    reducing token usage by 95% and preserving existing codebase integrity.
    """)

with st.expander("Q2: How does a Deep Agent maintain durability across server restarts during long-running tasks?"):
    st.markdown("""
    **Answer**: Through persistent state graph checkpointers (e.g. LangGraph with Postgres or SQLite). On every node execution,
    the complete state dictionary (active milestone, task DAG, scratchpad, thread ID) is serialized and committed to disk.
    If a server restarts or container dies, a worker can re-instantiate the graph using the exact thread ID and continue
    from the last uncommitted subtask without losing historical progress.
    """)

with st.expander("Q3: What is the difference between lossy and lossless context compression in agent runtimes?"):
    st.markdown("""
    **Answer**: Lossless compression preserves exact information while reducing representation size (e.g., parsing code into
    an AST and keeping only signatures, or removing duplicate terminal whitespace). Lossy compression uses an LLM to summarize
    conversational history or lengthy tool outputs into a concise prose overview, permanently discarding low-level tokens while
    retaining high-level semantic takeaways.
    """)

with st.expander("Q4: How does a Deep Agent know when a decomposed task has truly succeeded?"):
    st.markdown("""
    **Answer**: Deep Agents mandate deterministic verification rather than subjective LLM self-assessment. Each decomposed task
    must define an automated validation action: running `pytest` to assert passing tests, running a TypeScript compiler to check
    type errors, or validating that a newly written file exists and parses as valid JSON/YAML.
    """)

with st.expander("Q5: What are the risks of granting an autonomous agent filesystem write permissions?"):
    st.markdown("""
    **Answer**: Unrestricted write access can overwrite critical system files (`/etc/passwd`, `.env` secrets), delete git history,
    or introduce backdoors. Mitigations include: (1) Running inside a temporary container sandbox; (2) Restricting writes to an
    explicit project workspace path; (3) Enforcing git branch isolation so changes occur on a disposable branch; and (4) Human review
    before pushing to remote repositories.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Full-File Overwrite Truncation)**: Asking an LLM to re-generate an 800-line file when editing 3 lines. Models frequently truncate middle sections with `# ... rest of code unchanged`, destroying your file.")
    st.warning("**Trap 2 (Stateless Subagent Execution)**: Spawning subagents without passing critical project constraints (e.g., Python version, formatting guidelines), causing child agents to generate incompatible code.")
with col_trap2:
    st.error("**Trap 3 (In-Memory-Only State in Long Tasks)**: Keeping 3 hours of agent execution in Python RAM. A single network hiccup or OOM kill wipes out the entire session with zero recovery.")
    st.warning("**Trap 4 (Unbounded Output Buffers)**: Running a test suite or build command without stdout/stderr truncation limits, allowing millions of repetitive log tokens to flood and crash the agent's memory.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is Task Decomposition in Deep Agents?")
        st.caption("Flip for answer:")
        st.info("**Back**: The process of breaking a complex, open-ended goal into small, atomic, verifiable subtasks that can be executed and tested independently.")
    with st.container(border=True):
        st.markdown("**Front**: Why are Unified Diffs preferred over full file replacements?")
        st.caption("Flip for answer:")
        st.info("**Back**: Diffs only output modified lines, slashing token costs and preventing the model from accidentally dropping existing code.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is Context Compression?")
        st.caption("Flip for answer:")
        st.info("**Back**: Actively summarizing, truncating, and offloading tool observations to prevent memory saturation during long-running tasks.")
    with st.container(border=True):
        st.markdown("**Front**: How does a durable checkpointer enable multi-hour execution?")
        st.caption("Flip for answer:")
        st.info("**Back**: It persists state snapshots to a database on every turn, allowing the agent to pause, resume, and survive crashes across days.")

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand how Deep Agents decompose complex objectives into atomic subtasks.", value=True)
c2 = st.checkbox("I know how to design safe filesystem tools using surgical line-level diffs.", value=True)
c3 = st.checkbox("I understand the lifecycle of subagent spawning and garbage collection.", value=True)
c4 = st.checkbox("I know how to implement lossy and lossless context compression strategies.", value=True)
c5 = st.checkbox("I understand how durable checkpointers enable multi-day resumable tasks.", value=True)
c6 = st.checkbox("I recognize why deterministic verification (e.g. unit tests) is mandatory for milestones.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 27.3: Deep Agent Applications.")
