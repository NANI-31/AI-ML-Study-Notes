import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="27.3 Deep Agent Applications - LangChain Architecture",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for domain application telemetry
st.markdown("""
<style>
    .app-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-domain {
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
class DeepApplicationDomainSimulator:
    """
    Simulation engine showcasing autonomous deep execution across four primary
    enterprise domains: Coding, Research, Data Analysis, and Complex DevOps Automation.
    """
    DOMAINS = {
        "Coding Agent": {
            "focus": "Full-stack bug fixing, multi-file refactoring, unit test generation, Git PR creation",
            "tools": ["ripgrep", "read_file_chunk", "apply_unified_diff", "run_pytest", "git_commit"],
            "verification_metric": "100% Passing Unit Tests & Lint Checks",
            "autonomy_level": "Level 4 (Autonomous with Human PR Review)"
        },
        "Research Agent": {
            "focus": "Multi-source literature discovery, citation graph traversal, hypothesis validation",
            "tools": ["arxiv_search", "semantic_scholar", "web_scraper", "citation_verifier"],
            "verification_metric": "Zero Hallucinated Citations & Fact Consistency",
            "autonomy_level": "Level 3 (Autonomous Synthesis)"
        },
        "Data Analysis Agent": {
            "focus": "Automated warehouse querying, statistical anomaly detection, executive dashboard generation",
            "tools": ["sql_replica_query", "pandas_analyzer", "plotly_generator", "anomaly_detector"],
            "verification_metric": "Mathematical Schema & Range Consistency",
            "autonomy_level": "Level 4 (Autonomous Reporting)"
        },
        "DevOps Automation Agent": {
            "focus": "Kubernetes pod triage, database migration scripts, incident response automation",
            "tools": ["kubectl_read", "log_analyzer", "canary_deployer", "slack_alert"],
            "verification_metric": "Health Probe Status & Zero Downtime SLA",
            "autonomy_level": "Level 3 (HITL Gated Deployment)"
        }
    }

    def simulate_workflow(self, domain_name: str) -> Dict[str, Any]:
        spec = self.DOMAINS.get(domain_name, self.DOMAINS["Coding Agent"])
        return {
            "domain": domain_name,
            "focus": spec["focus"],
            "tools_deployed": spec["tools"],
            "verification": spec["verification_metric"],
            "autonomy": spec["autonomy_level"],
            "milestones_completed": 4,
            "status": "MISSION_SUCCESS"
        }


# --- LESSON HEADER ---
st.title("27.3 Deep Agent Applications: Real-World Enterprise Automation")
st.caption("Chapter 27: Deep Agents | Domain: 9. LangChain Orchestration & LCEL")
st.markdown("""
Deep Agents are transforming software engineering, scientific literature review, quantitative data analysis,
and autonomous enterprise IT operations. By combining durable state persistence, hierarchical planning, subagents,
and safe execution sandboxes, these agents independently resolve complex, multi-stage business challenges.

In this lesson, you will master:
1. Architecting **Coding Agents** capable of repository exploration, multi-file refactoring, and test-driven fixes.
2. Deploying **Research Agents** for automated literature discovery, source verification, and synthesized authoring.
3. Building **Data Analysis Agents** executing SQL queries, statistical models, and executive data stories.
4. Designing **Autonomous Workflows** with event-driven triggers and end-to-end task completion.
5. Implementing **Complex Task Automation** across enterprise systems with strict safety guardrails.
""")
st.divider()

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. End-to-End Architectural Blueprint: Enterprise Autonomous Agent Applications", divider="rainbow")
st.markdown("""
A production Deep Agent application coordinates domain specialists, sandbox tools, and human review gates
across a unified enterprise integration fabric.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    app_dot = """
    digraph DeepAppBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

        subgraph cluster_enterprise_ingress {
            label="1. Ingress & Mission Triggers";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            github_issue [label="GitHub Issue / Jira Ticket\\n'Bug: Auth token leakage in logging'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            scheduler_trigger [label="Cron / Data Pipeline Event\\n'Nightly Revenue Anomaly Audit'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_specialized_runtimes {
            label="2. Autonomous Deep Agent Runtimes";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            coding_core [label="Coding Agent Runtime\\nAST indexing, diff generation, pytest", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            research_core [label="Research Agent Runtime\\nPaper indexing, cross-citation analysis", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            data_core [label="Data Analysis Runtime\\nSQL query synthesis, statistical charts", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
        }

        subgraph cluster_deliverables {
            label="3. Verified Deliverables & Egress";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            git_pr [label="Tested Pull Request on GitHub\\nIncludes unit tests & reproduction script", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            report_doc [label="Synthesized Research Brief\\nVerified citations, zero hallucinations", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            human_approval [label="Human-in-the-Loop Gate\\nExecutive signoff before production deployment", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
        }

        github_issue -> coding_core [label="Dispatches Task"];
        scheduler_trigger -> data_core [label="Scheduled Trigger"];
        coding_core -> git_pr [label="Emits Code Patch"];
        research_core -> report_doc [label="Emits Technical Doc"];
        data_core -> report_doc [label="Emits Data Narrative"];
        git_pr -> human_approval [label="Awaits Peer Review"];
    }
    """
    st.graphviz_chart(app_dot, use_container_width=True)

# --- SECTION 2: CODING AGENTS ---
st.header("2. Coding Agents", divider="rainbow")
st.markdown("""
Autonomous **Coding Agents** (such as Devin, SWE-agent, and specialized engineering assistants)
represent the most sophisticated domain application of Deep Agents. They tackle real software engineering
issues from issue tracker to pull request.
""")

col_code1, col_code2 = st.columns(2)
with col_code1:
    st.markdown("""
    #### The 5-Stage Autonomous Coding Loop:
    1. **Repository Discovery**: Uses `grep_search` and AST parsers to locate relevant classes, files, and dependency definitions.
    2. **Reproduction Test**: Writes a standalone reproduction script or unit test that triggers the reported bug (verifying failure).
    3. **Surgical Implementation**: Applies minimal, targeted line-based diffs to the affected files.
    4. **Test-Driven Verification**: Executes `pytest` or compiler tools inside an isolated sandbox until the reproduction test passes.
    5. **Regression Audit & PR Creation**: Runs the full project regression suite to ensure zero collateral damage, then opens a Git branch and Pull Request.
    """)

with col_code2:
    with st.container(border=True):
        st.markdown("💻 **The Test-Driven Development (TDD) Anchor**")
        st.caption("A coding agent without automated unit tests is dangerously unanchored.")
        st.markdown("""
        ```
        ┌────────────────────────────────────────────────────────┐
        │ 1. Write Reproduction Test (Fails: Expected bug)       │
        ├────────────────────────────────────────────────────────┤
        │ 2. Apply Code Patch in Sandbox                         │
        ├────────────────────────────────────────────────────────┤
        │ 3. Execute Pytest -> PASS!                             │
        ├────────────────────────────────────────────────────────┤
        │ 4. Run Full Regression Suite -> PASS!                  │
        ├────────────────────────────────────────────────────────┤
        │ 5. Stage, Commit, and Open GitHub Pull Request         │
        └────────────────────────────────────────────────────────┘
        ```
        """)

# --- SECTION 3: RESEARCH AGENTS ---
st.header("3. Research Agents", divider="rainbow")
st.markdown("""
**Research Agents** autonomously investigate open-ended questions across scientific literature, market reports,
and technical documentation. Unlike a standard search tool that returns 5 snippets, a Research Agent iterates
recursively over citation graphs.
""")

res_dot = """
digraph ResearchAgentWorkflow {
    rankdir=LR;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    topic [label="Research Topic: 'Speculative Decoding in LLMs'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
    arxiv_search [label="1. Paper Search\\narxiv API & Semantic Scholar", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
    citation_crawl [label="2. Citation Traversal\\nFollows foundational references", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
    fact_audit [label="3. Fact & Math Audit\\nCross-checks empirical speedups", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
    whitepaper [label="4. Synthesized Whitepaper\\nFormal markdown with DOI citations", fillcolor="#065F46", fontcolor="#D1FAE5", color="#34D399"];

    topic -> arxiv_search [label="Formulate queries"];
    arxiv_search -> citation_crawl [label="Extract top 10 papers"];
    citation_crawl -> fact_audit [label="Extract benchmarks"];
    fact_audit -> whitepaper [label="Zero Hallucination Review"];
}
"""
st.graphviz_chart(res_dot, use_container_width=True)

# --- SECTION 4: DATA ANALYSIS AGENTS ---
st.header("4. Data Analysis Agents", divider="rainbow")
st.markdown("""
**Data Analysis Agents** bridge natural language and quantitative data warehouses. They translate executive inquiries
into optimized SQL, parse returned tabular matrices, compute statistical tests, and generate interactive charts.
""")

col_da1, col_da2 = st.columns(2)
with col_da1:
    st.markdown("""
    #### Autonomous Data Analytics Capabilities:
    - **Schema Discovery**: Inspects tables, foreign keys, and column distributions to understand warehouse structure.
    - **Self-Correcting SQL**: If a database returns a syntax error or missing column, the agent inspects the schema and self-corrects the query.
    - **Anomaly Detection**: Executes statistical routines (Z-scores, isolation forests) to identify anomalies without human prompting.
    - **Executive Narrative Generation**: Synthesizes tabular numbers into clear business recommendations.
    """)

with col_da2:
    with st.container(border=True):
        st.markdown("📊 **Data Analysis Flow**")
        st.code("""
# 1. Translate prompt to SQL with AST validation
# 2. Execute query on read-replica database
# 3. Load result into Pandas DataFrame
# 4. Compute statistical summaries
# 5. Generate interactive Plotly figure
# 6. Deliver narrative insight to stakeholder
        """, language="python")

# --- SECTION 5: AUTONOMOUS WORKFLOWS ---
st.header("5. Autonomous Workflows", divider="rainbow")
st.markdown("""
An **Autonomous Workflow** is a multi-agent system triggered by system events rather than immediate human interaction.
These pipelines run in background worker processes, monitoring telemetry, ingesting customer support backlogs,
and resolving maintenance tasks autonomously.
""")

st.subheader("Key Characteristics of Production Autonomous Workflows")
w_col1, w_col2, w_col3 = st.columns(3)
with w_col1:
    with st.container(border=True):
        st.markdown("⏰ **Event-Driven Triggers**")
        st.markdown("Activated by webhooks, message queues (Kafka, SQS), cron schedules, or incident monitoring alerts.")

with w_col2:
    with st.container(border=True):
        st.markdown("🛡️ **Bounded Authority**")
        st.markdown("Operates within strict budget and permission boundaries: can stage PRs and patch test environments, but cannot deploy to production without approval.")

with w_col3:
    with st.container(border=True):
        st.markdown("📑 **Comprehensive Auditing**")
        st.markdown("Logs every reasoning token, tool call, and file diff to LangSmith for post-hoc engineering review.")

# --- SECTION 6: COMPLEX TASK AUTOMATION ---
st.header("6. Complex Task Automation", divider="rainbow")
st.markdown("""
Complex task automation involves high-stakes multi-system tasks: cloud migrations, infrastructure reconciliation,
and compliance audits across multiple disparate enterprise platforms (AWS, Jira, Salesforce, GitHub).
""")

st.subheader("The Five Safety Commandments for Complex Task Automation")
st.markdown("""
1. **Commandment 1: Dry-Run First**: Before executing mutating commands (e.g., `terraform apply`, `kubectl delete`), the agent must run and present a dry-run plan.
2. **Commandment 2: Cryptographic Identity**: Every automated action must be signed with the agent's unique service account identity for complete audit attribution.
3. **Commandment 3: Human Break-Glass Gate**: Any action classified as HIGH or CRITICAL risk must halt for explicit human approval.
4. **Commandment 4: Automated Rollback**: Every deployment tool must be accompanied by an automated rollback procedure if post-flight health checks fail.
5. **Commandment 5: Hard Spending Ceilings**: Enforce maximum token, tool invocation, and cloud API budget caps per workflow run.
""")

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: End-to-End Autonomous Coding Agent", divider="rainbow")
st.markdown("""
Below is a runnable implementation of an Autonomous Coding Agent incorporating repository exploration,
reproduction test creation, surgical diffing, and automated test execution.
""")

prod_app_code = '''from typing import Dict, Any, List
import subprocess

class AutonomousCodingAgent:
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.scratchpad: List[str] = []

    def run_tool_grep(self, pattern: str) -> str:
        """Simulates searching the repository for a token."""
        return f"Found match in src/auth.py:L142: def verify_jwt(token):"

    def write_reproduction_test(self, test_code: str) -> str:
        """Writes a unit test to verify the bug."""
        return "Saved reproduction script to tests/test_bug_reproduce.py"

    def apply_patch(self, diff: str) -> str:
        """Applies a surgical line-based diff."""
        return "Applied unified diff to src/auth.py successfully."

    def run_tests(self) -> Dict[str, Any]:
        """Runs the unit test suite inside the execution sandbox."""
        # In production: subprocess.run(["pytest", "tests/"])
        return {"exit_code": 0, "passed": 12, "failed": 0, "status": "ALL_TESTS_PASS"}

    def execute_issue_resolution(self, issue_description: str) -> dict:
        self.scratchpad.append(f"Starting work on issue: {issue_description}")
        
        # Step 1: Repo Exploration
        grep_result = self.run_tool_grep("verify_jwt")
        self.scratchpad.append(grep_result)
        
        # Step 2: Write Reproduction Test
        test_res = self.write_reproduction_test("def test_expired_token(): assert verify_jwt(expired) is False")
        self.scratchpad.append(test_res)
        
        # Step 3: Apply Code Patch
        patch_res = self.apply_patch("+++ b/src/auth.py\\n+ if token.expired: return False")
        self.scratchpad.append(patch_res)
        
        # Step 4: Verify with Sandbox Test Execution
        test_run = self.run_tests()
        
        if test_run["exit_code"] == 0:
            return {
                "status": "SUCCESS",
                "deliverable": "Pull Request #412 ready for human review.",
                "verification": test_run["status"]
            }
        else:
            return {"status": "FAILED", "deliverable": "Tests failed. Initiating replan loop."}
'''
st.code(prod_app_code, language="python")

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Autonomous Agent Domain Dispatcher", divider="rainbow")
st.markdown("""
Select an enterprise domain to simulate how Deep Agents configure tools, establish verification metrics,
and execute multi-stage missions autonomously.
""")

domain_engine = DeepApplicationDomainSimulator()

col_aw1, col_aw2 = st.columns([1, 1])

with col_aw1:
    st.subheader("🎯 Enterprise Domain Selection")
    chosen_domain = st.selectbox(
        "Choose Deep Agent Application Domain:",
        list(domain_engine.DOMAINS.keys())
    )
    deploy_btn = st.button("🚀 Launch Autonomous Domain Mission", type="primary", use_container_width=True)

with col_aw2:
    st.subheader("📊 Autonomous Execution Telemetry")
    if deploy_btn:
        mission_res = domain_engine.simulate_workflow(chosen_domain)
        
        st.markdown(
            f"<div class='app-card'><span class='badge-domain'>{chosen_domain.upper()}</span>"
            f"<br><br><b>Mission Objective:</b> {mission_res['focus']}"
            f"<br><b>Autonomy Profile:</b> {mission_res['autonomy']}</div>",
            unsafe_allow_html=True
        )

        st.markdown("##### 🛠️ Tools Deployed in Sandbox")
        tool_tags = " ".join([f"`{t}`" for t in mission_res["tools_deployed"]])
        st.markdown(tool_tags)

        st.markdown("##### 🎯 Verification Standard")
        st.info(f"**Verification Criteria:** {mission_res['verification']}")
        st.success(f"✅ Mission Status: {mission_res['status']} — All 4 execution milestones completed cleanly.")
    else:
        st.info("Select a domain and click 'Launch Autonomous Domain Mission' to inspect tooling and verification telemetry.")

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: What prevents an autonomous coding agent from corrupting a production codebase?"):
    st.markdown("""
    **Answer**: Defense-in-depth across the execution pipeline:
    (1) **Sandbox Isolation**: Agents run inside disposable containers or ephemeral VMs without network access to production systems;
    (2) **Branch Isolation**: All modifications occur on dedicated git feature branches (`agent/issue-123`), never on `main`;
    (3) **Automated Test Enforcement**: The agent cannot declare success unless existing regression tests pass; and
    (4) **Human Pull Request Review**: The final pull request must be reviewed, approved, and merged by a human engineer.
    """)

with st.expander("Q2: How do Research Agents ensure that scientific citations are authentic and not hallucinated?"):
    st.markdown("""
    **Answer**: By decoupling citation discovery from prose generation. When generating claims, the Research Agent is forced
    to use a structured **Citation Verification Tool** that cross-checks paper titles, authors, and DOIs against live APIs
    (arXiv, CrossRef, Semantic Scholar). Any citation that does not return a verified DOI match is stripped before final synthesis.
    """)

with st.expander("Q3: Why are Data Analysis Agents more effective than simple text-to-SQL prompts?"):
    st.markdown("""
    **Answer**: Text-to-SQL prompts often fail on real-world queries due to missing domain knowledge or complex joins.
    Data Analysis Agents operate in an iterative loop: they discover schema metadata, execute sample queries to inspect actual values,
    catch database syntax errors and self-correct, compute statistical tests, and visualize data in interactive plots,
    delivering a complete analytical narrative rather than a raw SQL snippet.
    """)

with st.expander("Q4: What is the significance of the Reproduction Test in autonomous bug fixing?"):
    st.markdown("""
    **Answer**: The reproduction test proves the bug exists before any code is modified. If the reproduction test passes initially,
    the agent knows it has not reproduced the issue properly. Once code changes are made, the reproduction test serves as an objective,
    deterministic proof that the fix resolved the problem, eliminating reliance on the LLM's subjective opinion.
    """)

with st.expander("Q5: What are the primary failure modes of autonomous event-driven workflows in enterprise IT?"):
    st.markdown("""
    **Answer**: (1) **Cascading Automation Loops**: An agent creates an incident ticket, triggering an alert, which triggers the agent to create another ticket;
    (2) **Flaky Third-Party APIs**: A tool timeout causes the agent to assume an action failed and re-attempt it, causing duplicate operations;
    and (3) **Stale Context**: An agent operating on cached schema definitions that have changed since the mission started.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Autonomous Merging to Main)**: Permitting an agent to merge code directly to the primary branch without human review. Subtly flawed logic or security regressions can bypass automated test suites.")
    st.warning("**Trap 2 (Unbounded Research Crawling)**: Allowing a research agent to traverse citation links infinitely without link depth caps, consuming thousands of dollars in web-scraping and LLM costs.")
with col_trap2:
    st.error("**Trap 3 (Non-Idempotent Automation Tools)**: Tools that modify resources without idempotency keys. If a network blip occurs, re-running the tool can duplicate billing transactions or cloud resources.")
    st.warning("**Trap 4 (Omitting Health Probes After Changes)**: Marking a DevOps migration task complete immediately after executing an apply command without waiting for Kubernetes health probes to report healthy.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the Reproduction Test pattern in coding agents?")
        st.caption("Flip for answer:")
        st.info("**Back**: Writing a unit test that fails before code modification and passes only after the bug is fixed, providing objective proof of resolution.")
    with st.container(border=True):
        st.markdown("**Front**: How do Research Agents verify citation accuracy?")
        st.caption("Flip for answer:")
        st.info("**Back**: By validating every cited paper and DOI against external APIs (arXiv, CrossRef) before emitting the final text.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is an Event-Driven Autonomous Workflow?")
        st.caption("Flip for answer:")
        st.info("**Back**: A multi-agent pipeline activated automatically by webhooks, queue messages, or cron events rather than manual human prompts.")
    with st.container(border=True):
        st.markdown("**Front**: Why is Idempotency mandatory for autonomous agent tools?")
        st.caption("Flip for answer:")
        st.info("**Back**: Because tool retries after network timeouts must not produce duplicate side effects (e.g. double charging a customer).")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the 5-stage execution loop of autonomous coding agents.", value=True)
c2 = st.checkbox("I know how research agents enforce zero-hallucination citation verification.", value=True)
c3 = st.checkbox("I understand how data analysis agents combine SQL synthesis with statistical narrative.", value=True)
c4 = st.checkbox("I recognize why reproduction tests are mandatory for objective bug fixing.", value=True)
c5 = st.checkbox("I understand the safety commandments for complex enterprise task automation.", value=True)
c6 = st.checkbox("I know how to design sandbox and git branch isolation for autonomous operations.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You have completed Chapter 27: Deep Agents.")
