import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="25.2 Tool Security - LangChain Security Architecture",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for security telemetry and permission tables
st.markdown("""
<style>
    .perm-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-perm-allow {
        background-color: #065F46;
        color: #34D399;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-perm-deny {
        background-color: #7F1D1D;
        color: #F87171;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-perm-human {
        background-color: #78350F;
        color: #FBBF24;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class SecurityToolRegistry:
    """
    Simulation engine showcasing role-based tool gating, dynamic allowlists,
    and sandboxed execution telemetry.
    """
    TOOL_CATALOG = {
        "read_docs": {
            "name": "read_docs",
            "tier": "READ_ONLY",
            "required_role": "viewer",
            "risk_level": "LOW",
            "description": "Searches and reads internal non-sensitive documentation."
        },
        "query_sql_analytics": {
            "name": "query_sql_analytics",
            "tier": "READ_ONLY",
            "required_role": "analyst",
            "risk_level": "MEDIUM",
            "description": "Executes SELECT-only queries against read-replica warehouse."
        },
        "send_customer_email": {
            "name": "send_customer_email",
            "tier": "EXTERNAL_EFFECT",
            "required_role": "support_agent",
            "risk_level": "HIGH",
            "description": "Dispatches communication to verified customer addresses."
        },
        "execute_python_code": {
            "name": "execute_python_code",
            "tier": "CODE_EXEC",
            "required_role": "developer",
            "risk_level": "CRITICAL",
            "description": "Runs arbitrary Python code in an isolated microVM sandbox."
        },
        "drop_db_table": {
            "name": "drop_db_table",
            "tier": "DESTRUCTIVE",
            "required_role": "admin",
            "risk_level": "CRITICAL",
            "description": "Executes destructive schema migrations on production databases."
        }
    }

    ROLE_HIERARCHY = {
        "viewer": 1,
        "analyst": 2,
        "support_agent": 2,
        "developer": 3,
        "admin": 4
    }

    def evaluate_tool_call(self, tool_name: str, user_role: str, human_approved: bool = False) -> Dict[str, Any]:
        if tool_name not in self.TOOL_CATALOG:
            return {
                "allowed": False,
                "reason": f"Tool '{tool_name}' not in certified catalog allowlist.",
                "action": "BLOCKED"
            }

        tool_meta = self.TOOL_CATALOG[tool_name]
        req_role = tool_meta["required_role"]
        
        user_level = self.ROLE_HIERARCHY.get(user_role, 0)
        req_level = self.ROLE_HIERARCHY.get(req_role, 99)

        if user_level < req_level:
            return {
                "allowed": False,
                "reason": f"Access Denied: Role '{user_role}' lacks required permission tier '{req_role}'.",
                "action": "BLOCKED"
            }

        # Check for dangerous actions requiring human approval
        if tool_meta["risk_level"] == "CRITICAL" and not human_approved:
            return {
                "allowed": False,
                "reason": f"Tool '{tool_name}' is classified as CRITICAL. Requires explicit human confirmation.",
                "action": "AWAITING_HUMAN_APPROVAL"
            }

        return {
            "allowed": True,
            "reason": f"Permission Granted. User role '{user_role}' authorized for {tool_meta['tier']}.",
            "action": "EXECUTED",
            "details": tool_meta
        }


# --- LESSON HEADER ---
st.title("25.2 Tool Security & Zero-Trust Execution")
st.caption("Chapter 25: Security in LangChain | Domain: 9. LangChain Orchestration & LCEL")
st.markdown("""
In modern agentic architectures, language models are granted the ability to invoke external tools: querying databases,
calling REST APIs, sending notifications, and executing code. Unchecked tool execution is the single greatest security
vulnerability in production AI systems.

In this lesson, you will master:
1. Enforcing granular Tool Permissions based on user session roles (RBAC).
2. Implementing dynamic Tool Allowlists to prevent runtime privilege escalation.
3. Managing Dangerous Tools through mandatory human-in-the-loop approval gates.
4. Securing External API integrations against SSRF (Server-Side Request Forgery) and credential leakages.
5. Restricting Database Tools with read-only replicas, static SQL parsing, and row ceilings.
6. Isolating Code Execution tools in hardened containerized sandboxes and microVMs.
""")
st.divider()

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. End-to-End Architectural Blueprint: Zero-Trust Tool Execution Runtime", divider="rainbow")
st.markdown("""
A zero-trust tool architecture ensures that no tool is executed solely because an LLM emitted a tool-call JSON payload.
Every proposed tool invocation passes through an authorization proxy, an allowlist validator, and an execution sandbox.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    tool_sec_dot = """
    digraph ToolSecurityBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

        subgraph cluster_llm_generation {
            label="1. Agent Reasoning & Tool Request";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            llm_plan [label="Agent Reasoning\\nDecides to execute action", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            raw_tool_call [label="Proposed Tool Call\\n{'name': 'query_sql', 'args': {...}}", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
        }

        subgraph cluster_gateway {
            label="2. Zero-Trust Security Gateway";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            allowlist_filter [label="Tool Allowlist Check\\nIs tool authorized for tenant?", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            rbac_validator [label="RBAC Session Check\\nDoes current JWT possess role?", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            hitl_gate [label="Human-in-the-Loop Intercept\\nFlag if risk_level == CRITICAL", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
        }

        subgraph cluster_sandbox {
            label="3. Isolated Execution Runtime";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            read_sql [label="Read-Only SQL Replica\\nAST Parser blocks DROP/UPDATE", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            microvm_box [label="Sandboxed Container (gVisor)\\nNo network, 500ms timeout", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            safe_egress [label="Sanitized Result Output\\nRedacted PII passed back to Agent", fillcolor="#065F46", fontcolor="#D1FAE5", color="#34D399"];
        }

        llm_plan -> raw_tool_call [label="Emits ToolCall"];
        raw_tool_call -> allowlist_filter [label="Intercepted by Proxy"];
        allowlist_filter -> rbac_validator [label="In Catalog"];
        rbac_validator -> hitl_gate [label="Authorized User"];
        hitl_gate -> read_sql [label="Approved (DB Call)"];
        hitl_gate -> microvm_box [label="Approved (Code Run)"];
        read_sql -> safe_egress [label="Data Filtered"];
        microvm_box -> safe_egress [label="Execution Result"];
    }
    """
    st.graphviz_chart(tool_sec_dot, use_container_width=True)

# --- SECTION 2: TOOL PERMISSIONS ---
st.header("2. Tool Permissions", divider="rainbow")
st.markdown("""
In multi-user enterprise systems, different users possess different rights. A regular support representative should never
trigger database migration tools or execute shell scripts. Tool permissions bind the runtime execution of tools to the
end-user's authenticated identity and session context.
""")

col_perm1, col_perm2 = st.columns(2)
with col_perm1:
    st.markdown("""
    #### Core Implementation Principles
    - **Context Propagation**: Always forward the active user's authorization claims (e.g., JWT claims, tenant ID) inside LangChain `RunnableConfig["configurable"]`.
    - **Dynamic Tool Binding**: Instead of binding all available tools to `model.bind_tools()`, filter the tools list before binding so unauthorized tools are invisible to the model.
    - **Double-Gate Authorization**: Validate permissions at prompt-binding time AND again at tool execution time to prevent adversarial token hallucinations.
    """)

with col_perm2:
    with st.container(border=True):
        st.markdown("🔑 **Dynamic Tool Filter Pattern**")
        st.code("""
# Filter available tools by active user role
def get_tools_for_user(user_role: str):
    all_tools = [search_kb, query_sql, run_code, refund_user]
    if user_role == "customer":
        return [search_kb]
    elif user_role == "support":
        return [search_kb, refund_user]
    elif user_role == "admin":
        return all_tools
    return []

# Bind only permitted tools to ChatModel
user_tools = get_tools_for_user(session["role"])
model_with_tools = model.bind_tools(user_tools)
        """, language="python")

# --- SECTION 3: TOOL ALLOWLISTS ---
st.header("3. Tool Allowlists", divider="rainbow")
st.markdown("""
A Tool Allowlist is an immutable catalog defining explicitly authorized tools. An agent runtime must reject any tool
name that is not explicitly registered on the active allowlist.
""")

st.subheader("Comparison: Static Open Tools vs Modern Dynamic Allowlists")
comp_allow = pd.DataFrame({
    "Security Dimension": ["Catalog Scope", "Input Validation", "Execution Gating", "Compromise Blast Radius"],
    "Unrestricted Agent (Antipattern)": ["Exposes all internal tools globally", "Model generates arbitrary args", "Immediate automatic execution", "Total system compromise"],
    "Dynamic Tool Allowlist (Best Practice)": ["Scoped per tenant and user role", "Strict Pydantic schema validation", "Pre-execution authorization gateway", "Restricted to granted tool subset"]
})
st.dataframe(comp_allow, use_container_width=True)

# --- SECTION 4: DANGEROUS TOOLS ---
st.header("4. Dangerous Tools", divider="rainbow")
st.markdown("""
Dangerous tools are functions that cause side effects in the external world: sending emails, updating records,
modifying file systems, transferring funds, or provisioning infrastructure.
""")

st.subheader("The Three-Tier Tool Classification Matrix")
c_tier1, c_tier2, c_tier3 = st.columns(3)
with c_tier1:
    with st.container(border=True):
        st.markdown("🟢 **Tier 1: Read-Only (Autonomous)**")
        st.caption("Risk: LOW")
        st.markdown("""
        - Read knowledge base articles
        - Fetch non-sensitive product docs
        - Query read-only cache
        - **Execution**: Fully automated
        """)

with c_tier2:
    with st.container(border=True):
        st.markdown("🟡 **Tier 2: Stateful / Scoped (Audit Logged)**")
        st.caption("Risk: MEDIUM")
        st.markdown("""
        - Update user preferences
        - Create draft support ticket
        - Schedule calendar invite
        - **Execution**: Automated with mandatory audit trail
        """)

with c_tier3:
    with st.container(border=True):
        st.markdown("🔴 **Tier 3: Destructive / Financial (HITL Gate)**")
        st.caption("Risk: CRITICAL")
        st.markdown("""
        - Execute refund or billing charge
        - Delete database records
        - Execute terminal code
        - **Execution**: Mandatory Human-in-the-Loop Approval
        """)

# --- SECTION 5: EXTERNAL API SECURITY ---
st.header("5. External API Security", divider="rainbow")
st.markdown("""
When tools invoke external REST APIs or webhooks, applications face severe egress threats such as **Server-Side Request Forgery (SSRF)**,
credential theft, and resource exhaustion.
""")

api_sec_dot = """
digraph APISecurityFlow {
    rankdir=LR;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    agent_call [label="Agent Webhook Tool\\nURL: http://169.254.169.254/latest/meta-data/", fillcolor="#7F1D1D", fontcolor="#FCA5A5", color="#EF4444"];
    egress_proxy [label="Egress Security Proxy\\nBlocks private IP ranges (10.x, 169.254.x)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
    auth_vault [label="Secrets Vault\\nInjects scoped API token", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
    external_api [label="External Verified Service\\n(e.g., Stripe, Sendgrid)", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];

    agent_call -> egress_proxy [label="Outbound Request"];
    egress_proxy -> auth_vault [label="Passes SSRF Check"];
    auth_vault -> external_api [label="Signed HTTPS Call"];
}
"""
st.graphviz_chart(api_sec_dot, use_container_width=True)

st.markdown("""
#### Production Hardening Measures for API Tools:
1. **SSRF Blocking**: Explicitly prohibit tools from connecting to loopback (`127.0.0.1`), private RFC1918 subnets (`10.0.0.0/8`, `192.168.0.0/16`), and cloud metadata services (`169.254.169.254`).
2. **Egress Allowlisting**: Tools must only communicate with approved external domain names over HTTPS.
3. **No Direct Secret Sharing**: Never expose API keys to the LLM context. Tools receive keys inside server-side closures.
""")

# --- SECTION 6: DATABASE SECURITY ---
st.header("6. Database Security", divider="rainbow")
st.markdown("""
Natural Language to SQL tools are susceptible to SQL injection and unintended bulk data exfiltration.
A production SQL tool must never connect directly to primary write databases with administrative privileges.
""")

col_db1, col_db2 = st.columns(2)
with col_db1:
    st.markdown("""
    #### Database Gating Best Practices
    - **Dedicated Read-Only User**: Connect with a database role that has strict `SELECT`-only privileges on designated views.
    - **AST Static Query Analysis**: Before query execution, parse the generated SQL with a parser (e.g., `sqlglot`) to verify the absence of `DROP`, `DELETE`, `ALTER`, or `INSERT` statements.
    - **Hard Row Limits**: Force append `LIMIT 50` to all generated queries to prevent memory denial-of-service from massive tables.
    - **Table and Column Masking**: Restrict metadata inspection schemas so the agent cannot see passwords, salts, or credit card columns.
    """)

with col_db2:
    with st.container(border=True):
        st.markdown("🛡️ **AST-Based SQL Validator**")
        st.code("""
import sqlglot
from sqlglot.expressions import Delete, Drop, Update, Insert

def validate_sql_safety(query: str) -> bool:
    try:
        parsed = sqlglot.parse_one(query)
        forbidden_types = (Delete, Drop, Update, Insert)
        for node in parsed.walk():
            if isinstance(node, forbidden_types):
                raise ValueError("Destructive DDL/DML query rejected!")
        return True
    except Exception as e:
        raise ValueError(f"SQL Security Violation: {e}")
        """, language="python")

# --- SECTION 7: CODE EXECUTION SECURITY ---
st.header("7. Code Execution Security", divider="rainbow")
st.markdown("""
Python execution tools (such as Python REPL tools) allow models to execute dynamic code. Running arbitrary LLM-generated code
directly in the host application's operating system environment is catastrophic.
""")

st.subheader("Comparison of Code Sandboxing Technologies")
sandbox_df = pd.DataFrame({
    "Isolation Technology": ["Local Subprocess (exec())", "Docker Container", "gVisor / Firecracker MicroVM", "WebAssembly (WASM / Pyodide)"],
    "Security Boundary": ["None (Shared OS and filesystem)", "Linux Cgroups & Namespaces", "Independent kernel virtualization", "Browser/V8 Memory Sandbox"],
    "Host Escape Risk": ["CRITICAL (Trivial host compromise)", "LOW (Privilege escalation risk)", "NEAR-ZERO (Hypervisor boundary)", "ZERO (Pure memory isolation)"],
    "Startup Latency": ["< 1 ms", "~ 500 ms", "~ 100 ms", "< 10 ms"],
    "Production Readiness": ["FORBIDDEN in Production", "Acceptable for internal tools", "Gold Standard for enterprise LLMs", "Ideal for client-side evaluation"]
})
st.dataframe(sandbox_df, use_container_width=True)

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Secure Tool Execution Gateway", divider="rainbow")
st.markdown("""
Below is a production-grade secure tool dispatcher demonstrating pre-execution authorization checks,
Pydantic argument validation, and human-in-the-loop exception handling.
""")

prod_tool_code = '''from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from typing import Dict, Any, Callable

class SafeSQLInput(BaseModel):
    query: str = Field(description="SELECT-only query string to execute against analytics.")

class SecureToolDispatcher:
    def __init__(self, user_role: str):
        self.user_role = user_role
        self.role_permissions = {
            "viewer": ["read_docs"],
            "analyst": ["read_docs", "safe_sql_query"],
            "admin": ["read_docs", "safe_sql_query", "trigger_payout"]
        }

    def execute_tool_call(self, tool_name: str, tool_args: Dict[str, Any], tool_callable: Callable) -> Dict[str, Any]:
        # 1. Verify Allowlist and RBAC permissions
        allowed_tools = self.role_permissions.get(self.user_role, [])
        if tool_name not in allowed_tools:
            return {
                "status": "ERROR",
                "message": f"Security Violation: Role '{self.user_role}' unauthorized to execute '{tool_name}'."
            }

        # 2. Check for Critical Actions Requiring Human Confirmation
        if tool_name == "trigger_payout":
            return {
                "status": "INTERRUPT",
                "message": "Tool execution halted: 'trigger_payout' requires explicit Human-in-the-Loop signature."
            }

        # 3. Execute in Safe Environment
        try:
            result = tool_callable(**tool_args)
            return {"status": "SUCCESS", "result": result}
        except Exception as e:
            return {"status": "ERROR", "message": f"Execution failure: {str(e)}"}

# Example Safe Tool with Bound Pydantic Schema
@tool(args_schema=SafeSQLInput)
def safe_sql_query(query: str) -> str:
    """Executes a validated read-only SQL query."""
    if any(keyword in query.upper() for keyword in ["DROP", "DELETE", "UPDATE", "INSERT"]):
        raise ValueError("Mutating SQL queries are strictly prohibited.")
    return f"Simulated Query Result for: {query}"
'''
st.code(prod_tool_code, language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Tool Permission Gateway & Threat Simulator", divider="rainbow")
st.markdown("""
Simulate an agent attempting to execute various tools across different user authentication tiers.
Observe how the security gateway permits, blocks, or halts execution for human approval.
""")

registry = SecurityToolRegistry()

col_gw1, col_gw2 = st.columns([1, 1])

with col_gw1:
    st.subheader("👤 User Session & Tool Dispatch Configuration")
    user_role_select = st.selectbox(
        "Current Authenticated User Role:",
        ["viewer", "analyst", "support_agent", "developer", "admin"]
    )
    selected_tool = st.selectbox(
        "Proposed Agent Tool Call:",
        list(registry.TOOL_CATALOG.keys())
    )
    human_approved_toggle = st.toggle("Human Approval Granted (For Critical Tools)", value=False)
    dispatch_btn = st.button("⚡ Dispatch Tool Execution Call", type="primary", use_container_width=True)

with col_gw2:
    st.subheader("🛡️ Security Gateway Decision Engine")
    if dispatch_btn:
        decision = registry.evaluate_tool_call(selected_tool, user_role_select, human_approved_toggle)
        
        if decision["action"] == "BLOCKED":
            st.markdown(
                f"<div class='perm-card'><span class='badge-perm-deny'>EXECUTION BLOCKED</span>"
                f"<br><br><b>Reason:</b> {decision['reason']}</div>",
                unsafe_allow_html=True
            )
            st.error("🚨 Zero-Trust Policy Enforced: Unauthorized action suppressed.")
        elif decision["action"] == "AWAITING_HUMAN_APPROVAL":
            st.markdown(
                f"<div class='perm-card'><span class='badge-perm-human'>HUMAN-IN-THE-LOOP HALT</span>"
                f"<br><br><b>Status:</b> Execution paused. Critical tool invocation requires human signature.</div>",
                unsafe_allow_html=True
            )
            st.warning("⚠️ Critical Tool Intercepted: Approval required before state change occurs.")
        else:
            st.markdown(
                f"<div class='perm-card'><span class='badge-perm-allow'>EXECUTION AUTHORIZED</span>"
                f"<br><br><b>Tool Tier:</b> {decision['details']['tier']}"
                f"<br><b>Risk Rating:</b> {decision['details']['risk_level']}</div>",
                unsafe_allow_html=True
            )
            st.success(f"✅ Tool '{selected_tool}' executed inside sandboxed runtime.")
    else:
        st.info("Select a user role and tool, then click 'Dispatch Tool Execution Call' to evaluate.")

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: Why is it dangerous to let an LLM execute SQL queries directly on a production database?"):
    st.markdown("""
    **Answer**: Even with prompt instructions like 'Only generate SELECT queries', adversarial user prompts or hallucinated
    queries can generate `DROP TABLE`, `DELETE`, or `UPDATE` statements. In production, SQL agents must connect to isolated
    read-only replicas, enforce database-level user permissions forbidding mutations, pass generated SQL through AST parsers
    to verify query structure, and enforce hard `LIMIT` clauses to prevent memory exhaustion.
    """)

with st.expander("Q2: How does Server-Side Request Forgery (SSRF) threaten agents with API/Webhook tools?"):
    st.markdown("""
    **Answer**: If an agent is equipped with a tool that fetches URLs or sends webhooks, an attacker can manipulate the agent into
    requesting internal infrastructure endpoints (e.g., `http://169.254.169.254/latest/meta-data/` on AWS or internal private IPs).
    This can expose cloud credentials, internal Kubernetes APIs, or private intranet databases. Defense requires strict egress proxies
    blocking private IP ranges and enforcing domain allowlists.
    """)

with st.expander("Q3: What is the recommended strategy for sandboxing Python code execution tools in LangChain?"):
    st.markdown("""
    **Answer**: Never use Python's built-in `exec()` or uncontained local subprocesses. The industry gold standard is running
    code inside microVMs (e.g., AWS Firecracker, Fly.io machines, or gVisor container runtimes) with strict cgroups memory caps,
    disabled external networking, read-only temporary filesystems, and strict execution timeouts (e.g., 500ms to 2s).
    """)

with st.expander("Q4: How do dynamic tool allowlists improve both security and agent reasoning accuracy?"):
    st.markdown("""
    **Answer**: Dynamic allowlisting filters the tool schemas provided to `model.bind_tools()` based on the user's role and conversational
    state. Security-wise, it prevents unauthorized privilege escalation because the model is physically unaware of tools the user cannot use.
    Accuracy-wise, presenting 4 relevant tools instead of 40 global tools reduces distraction, eliminates tool hallucination, and conserves context tokens.
    """)

with st.expander("Q5: What role does Human-in-the-Loop (HITL) play in tool security?"):
    st.markdown("""
    **Answer**: For high-consequence irreversible tools (financial transfers, database deletes, mass emails), automated execution
    violates least privilege. Modern frameworks like LangGraph employ state interrupts (`interrupt()`) that pause graph execution,
    surface the proposed tool arguments to a human reviewer in an administrative UI, and only resume execution upon cryptographically
    verified approval.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Relying Solely on Prompt Instructions for Tool Safety)**: Prompting the model 'Do not call dangerous tools unless verified' is easily bypassed via jailbreaks. Security boundaries must be enforced in deterministic Python code.")
    st.warning("**Trap 2 (Hardcoding API Keys in Tool Docstrings)**: Placing sample API keys or internal endpoints inside tool description strings exposes secrets to the prompt token stream and downstream logs.")
with col_trap2:
    st.error("**Trap 3 (Unbounded Tool Execution Loops)**: Allowing an agent to call tools in an infinite loop without iteration caps allows denial-of-wallet attacks where an attacker exhausts API quotas via looping tool errors.")
    st.warning("**Trap 4 (Omitting Parameter Validation)**: Calling external tools without validating arguments via Pydantic models allows malformed inputs or buffer overflow attacks against legacy downstream microservices.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the Principle of Least Privilege in Agent Tool Calling?")
        st.caption("Flip for answer:")
        st.info("**Back**: An agent should only be provisioned with the minimal set of tools and database permissions necessary to complete the specific user's authorized request.")
    with st.container(border=True):
        st.markdown("**Front**: How does SSRF manifest in agent tools?")
        st.caption("Flip for answer:")
        st.info("**Back**: When an agent requests attacker-supplied URLs that resolve to internal private IP addresses or cloud metadata endpoints.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: Why is gVisor or Firecracker preferred over standard Docker for code execution?")
        st.caption("Flip for answer:")
        st.info("**Back**: Standard Docker containers share the host kernel. MicroVMs and gVisor implement independent guest kernels, preventing container breakout attacks.")
    with st.container(border=True):
        st.markdown("**Front**: What is dynamic tool filtering in LangChain?")
        st.caption("Flip for answer:")
        st.info("**Back**: Filtering the tools list bound to `model.bind_tools()` at runtime based on the authenticated user's session claims and permissions.")

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand why zero-trust gating is required for all agent tool invocations.", value=True)
c2 = st.checkbox("I know how to implement dynamic tool allowlisting scoped to user session roles.", value=True)
c3 = st.checkbox("I understand the three-tier tool risk classification matrix.", value=True)
c4 = st.checkbox("I know how to protect API tools against SSRF and credential leaks.", value=True)
c5 = st.checkbox("I understand AST query parsing and read-only replica requirements for SQL tools.", value=True)
c6 = st.checkbox("I know why containerized microVM sandboxes are mandatory for Python code execution.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 25.3: Data Security.")
