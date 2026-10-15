import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="22.4 Agent Security - Generative AI Curriculum",
    page_icon="🤖",
    layout="wide"
)

# Title & Overview Block
st.title("22.4 Agent Security: Tool Authorization, Sandboxing, Code Execution, & Human Approval")
st.caption("Chapter 22: LLM Safety and Responsible AI | Domain: 2. Large Language Models")

st.markdown("""
Autonomous AI Agents represent a transformative paradigm shift: models are no longer passive conversational engines; 
they are active computational orchestrators endowed with **Tool Calling** capabilities to execute arbitrary shell commands, 
query production SQL databases, communicate across public networks, and modify internal filesystem assets.

Equipping probabilistic next-token predictors with real-world execution side-effects creates critical enterprise hazards: 
**Tool Privilege Escalation**, **Arbitrary Remote Code Execution (RCE)**, **Data Exfiltration**, and **Unbounded Financial Resource Consumption**. 
Hardening production agents demands a defense-in-depth security perimeter: enforcing **Least Privilege Tool Authorization**, 
quarantining dynamic scripts within **Ephemeral Container Sandboxes (gVisor/Firecracker)**, applying **AST Code Parsing**, 
mandating **Human-in-the-Loop (HITL) Approval Gates** for destructive actions, and logging every intermediate thought and action to **Immutable Audit Trails**.
""")

st.markdown("""
In this lesson, you will master:
1. The unique attack surface of autonomous tool-using LLMs and ReAct loops.
2. Least Privilege Tool Authorization and tokenized scope delegation (OAuth/IAM capability tokens).
3. Sandboxing technologies: microVMs (Firecracker), container isolation (gVisor), and network egress air-gapping.
4. Python AST pre-execution static analysis and deterministic resource quotas (CPU, memory, wall-clock timeout).
5. Human-in-the-Loop (HITL) interruption architecture and immutable cryptographic audit logging for forensic compliance.
""")

st.divider()

# ==============================================================================
# SECTION 1: BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Hardened Autonomous Agent Security Architecture", divider="rainbow")

st.markdown("""
An enterprise agent security gateway implements an execution perimeter between the LLM planner and backend infrastructure:
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    agent_sec_dot = """
    digraph AgentSecurityArchitecture {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155", fontsize=11, margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10, penwidth=1.5];

        subgraph cluster_reasoning {
            label="Tier 1: Agent Reasoning & Action Proposal (Unprivileged)";
            style="rounded";
            bgcolor="#151E32";
            color="#38BDF8";
            fontcolor="#38BDF8";
            fontsize=12;

            UserPrompt [label="User Task Ingress\\n'Audit server logs and purge stale records'", fillcolor="#1E293B", color="#38BDF8"];
            AgentPlanner [label="Foundation LLM Reasoner\\nGenerates ReAct Action & Tool Call Proposal", fillcolor="#4338CA", color="#818CF8"];
            ProposedAction [label="Proposed Tool Invocation Payload\\nTool: 'db_execute', Args: {'sql': 'DROP TABLE...'}", fillcolor="#334155", color="#64748B"];

            UserPrompt -> AgentPlanner -> ProposedAction;
        }

        subgraph cluster_gateway {
            label="Tier 2: Policy Enforcement Gateway & HITL Interception";
            style="rounded";
            bgcolor="#151E32";
            color="#8B5CF6";
            fontcolor="#8B5CF6";
            fontsize=12;

            PolicyEngine [label="Tool Authorization & IAM Evaluator\\nVerifies Tool Whitelist & Scoped Capability Token", fillcolor="#0F766E", color="#14B8A6"];
            ASTAnalyzer [label="Static Code & AST Parser\\nRejects eval(), socket, exec(), os.system()", fillcolor="#6D28D9", color="#A78BFA"];
            RiskClassifier [label="Action Risk Classifier\\nRead-Only = Low Risk | Drop/Update = High Risk", fillcolor="#0369A1", color="#38BDF8"];
            HITLGate [label="Human-in-the-Loop Approval Queue\\nRequires multi-factor operator signature for High Risk", fillcolor="#F59E0B", color="#FCD34D", fontcolor="#000000"];

            ProposedAction -> PolicyEngine -> ASTAnalyzer -> RiskClassifier;
            RiskClassifier -> HITLGate [label="High Risk Action Detected", color="#F59E0B"];
        }

        subgraph cluster_sandbox {
            label="Tier 3: Isolated Sandboxed Execution & Audit Logger";
            style="rounded";
            bgcolor="#151E32";
            color="#10B981";
            fontcolor="#10B981";
            fontsize=12;

            SandboxEnv [label="Ephemeral MicroVM Sandbox (Firecracker / gVisor)\\nAir-gapped network, 256MB RAM cap, 5s timeout", fillcolor="#1E293B", color="#38BDF8", shape=cylinder];
            ExecutionResult [label="Sanitized Execution Observation\\nTruncated stdout/stderr output", fillcolor="#047857", color="#10B981"];
            AuditLogger [label="Append-Only Tamper-Proof Audit Store\\nSHA-256 Merkle log of Prompt + Action + Result", fillcolor="#065F46", color="#34D399"];

            HITLGate -> SandboxEnv [label="Approved by Operator", color="#10B981"];
            RiskClassifier -> SandboxEnv [label="Low Risk Auto-Exec", color="#38BDF8"];
            SandboxEnv -> ExecutionResult -> AuditLogger;
            ExecutionResult -> AgentPlanner [label="Observation Fed Back into ReAct Loop", style="dashed", color="#38BDF8"];
        }

        HITLGate -> ProposedAction [label="Rejected by Operator -> Abort", style="dotted", color="#EF4444", fontcolor="#EF4444"];
    }
    """
    st.graphviz_chart(agent_sec_dot, use_container_width=True)

st.markdown("""
### The Autonomous Agent Risk Spectrum
The table below illustrates the escalation of operational risks when granting LLMs autonomous tool execution:
""")

agent_risk_table_df = pd.DataFrame({
    "Tool Category": ["1. Read-Only Retrieval (Search, SQL Select)", "2. Network Egress (HTTP GET/POST, APIs)", "3. Filesystem Modification (File Write/Delete)", "4. Arbitrary Code Execution (Python, Bash, CLI)"],
    "Operational Hazard": ["Information disclosure, internal database schema leakage", "Data exfiltration via webhooks, Server-Side Request Forgery (SSRF)", "Accidental deletion of production records, ransomware-style overwrite", "Remote Code Execution (RCE), container breakout, cryptomining"],
    "Mandatory Security Control": ["Column-level view restrictions, read-only replica connection strings", "Strict outbound IP/domain allowlists, DNS filtering, air-gapped sandbox", "Write-quarantined ephemeral scratch disk, path traversal sanitization", "MicroVM virtualization (Firecracker), seccomp-bpf system call filters, strict AST validation"],
    "HITL Required?": ["Automated Execution (Low Risk)", "Conditional (Auto for allowlisted domains, HITL for new hosts)", "Mandatory HITL for deletions or updates outside `/tmp`", "Mandatory HITL for unverified shell commands"]
})
st.dataframe(agent_risk_table_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - TOOL AUTHORIZATION
# ==============================================================================
st.header("2. Tool Authorization: Least Privilege Scoping, Granular IAM, and Capability Tokens", divider="rainbow")

st.markdown("""
In many naive agent implementations, the model is given access to broad tools with unrestricted master API keys. 
**Tool Authorization** enforces the principle of least privilege: an agent should possess only the absolute minimum 
permissions required to accomplish its immediate step.
""")

col_ta1, col_ta2 = st.columns([1, 1])
with col_ta1:
    st.markdown("""
    ### Granular Capability Tokens
    * Instead of granting an agent static master database credentials, the gateway issues an **ephemeral capability token** scoped to a single query:
      `Token Scope: {"tool": "db_query", "action": "SELECT", "table": "customers", "expires_in": "30s"}`
    * Once the tool executes or the timeout expires, the token is permanently invalidated.
    * If an injection attack tricks the model into executing `DROP TABLE`, the database rejects the command at the credential authorization layer.
    """)

with col_ta2:
    st.markdown("""
    ### Schema-Level Validation
    * Defining tools with strict JSON Schemas (Pydantic models) with regex constraints on arguments:
      ```python
      class DBQueryArgs(BaseModel):
          query_type: Literal["SELECT"]
          limit: int = Field(le=100)
          table: Literal["inventory", "orders"]
      ```
    * Rejecting raw free-form string parameters (`sql_string: str`) in favor of parameterized schemas eliminates SQL injection vectors at the argument parser.
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - SANDBOXING
# ==============================================================================
st.header("3. Sandboxing: Ephemeral Container Isolation, gVisor, and Virtual Environments", divider="rainbow")

st.markdown("""
When an agent is permitted to write and execute code (such as data analysis with Python, Pandas, and Matplotlib), 
executing that code on the host operating system is catastrophic. **Sandboxing** isolates the execution environment 
so that any malicious command cannot access the host filesystem, network, or kernel.
""")

sandbox_comparison_df = pd.DataFrame({
    "Isolation Technology": ["Standard Docker Container", "gVisor (runsc) Sandbox", "Firecracker MicroVM", "WebAssembly (WASM / Pyodide)"],
    "Isolation Mechanism": ["Linux namespaces and cgroups; shares host Linux kernel", "Userspace kernel intercepts and re-implements Linux syscalls", "Minimalist hardware-assisted KVM virtualization (< 5MB footprint)", "Browser/runtime bytecode sandbox; zero OS syscall access"],
    "Security Level": ["Moderate (Vulnerable to Linux kernel CVE privilege escalation)", "High (Strong kernel barrier; hostile syscalls cannot breach host)", "Maximum (Hardware-enforced hypervisor boundary)", "High (Strict in-memory isolation; network blocked by default)"],
    "Startup Latency": ["500ms - 2,000ms", "50ms - 200ms", "5ms - 15ms (Ultra-fast boot)", "1ms - 5ms (Instant execution in-process)"],
    "Production Use Case": ["Development prototypes only", "Enterprise production multi-tenant code execution", "Frontier AI code sandboxes (OpenAI Code Interpreter, AWS Lambda)", "Client-side browser code evaluation"]
})
st.dataframe(sandbox_comparison_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - CODE EXECUTION SAFETY
# ==============================================================================
st.header("4. Code Execution Safety: AST Pre-Execution Parsing, Resource Quotas, and Air-Gapping", divider="rainbow")

st.markdown("""
Before transmitting code to the sandbox, a secure agent architecture evaluates the script through **Static AST (Abstract Syntax Tree) Analysis** 
and imposes strict runtime resource boundaries.
""")

col_ce1, col_ce2 = st.columns([1, 1])
with col_ce1:
    st.markdown("""
    ### AST Node Inspection Rules
    Before execution, the Python AST is traversed to reject prohibited nodes:
    1. **Banned Function Calls**: Blocking `eval()`, `exec()`, `compile()`, `__import__()`.
    2. **Banned Module Imports**: Rejecting `os`, `sys`, `subprocess`, `socket`, `shutil`, `ctypes`.
    3. **Attribute Access Restrictions**: Blocking private dunder attributes (`__subclasses__`, `__globals__`, `__code__`) used in Python sandbox escape exploits.
    """)

with col_ce2:
    st.markdown("""
    ### Deterministic Resource Quotas
    * **Execution Timeout (Wall-clock)**: Enforcing a hard timeout (e.g., 5 seconds) to prevent infinite loops (`while True: pass`).
    * **Memory Ceiling**: Capping VRAM and RAM (e.g., 256MB) via Linux cgroups to prevent memory exhaustion (OOM crashes).
    * **Disk Quota**: Restricting write operations to an in-memory `/tmp` filesystem capped at 50MB.
    * **Network Air-Gap**: Disabling all outbound network interfaces (`--net=none`) to ensure the agent cannot transmit stolen data.
    """)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - API SECURITY
# ==============================================================================
st.header("5. API Security: Secret Vaults, Mutual TLS, and Ephemeral Scoped Access", divider="rainbow")

st.markdown("""
When agents interact with third-party web APIs (Salesforce, Stripe, GitHub, Jira), the architecture must safeguard 
API credentials against prompt extraction and tool argument manipulation.
""")

st.markdown("""
### Architectural Guidelines for Agent API Keys
1. **Never Expose Raw Keys to Prompt Context**: The LLM should never be told the API token. The agent merely outputs a high-level function call:
   `call_stripe_refund(charge_id="ch_123", amount=50)`
   The backend API gateway intercepts the call, authenticates the agent session, retrieves the secret from a secure vault (HashiCorp Vault / AWS Secrets Manager), and injects the authorization header.
2. **Mutual TLS (mTLS)**: Enforcing encrypted, mutually authenticated tunnels between the agent execution engine and internal microservices.
3. **Idempotency Keys**: Attaching unique UUID idempotency keys to financial or transactional API calls to prevent duplicate billing if the ReAct loop retries an action.
""")

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - HUMAN APPROVAL
# ==============================================================================
st.header("6. Human Approval: Human-in-the-Loop (HITL) Interception for High-Stakes Tool Actions", divider="rainbow")

st.markdown("""
Autonomous ReAct loops can easily enter pathological states when faced with unexpected tool errors. 
**Human-in-the-Loop (HITL)** architecture ensures that actions exceeding a predetermined risk threshold are **paused** 
until an authenticated human operator signs off on the execution.
""")

col_hitl1, col_hitl2 = st.columns([1, 1])
with col_hitl1:
    st.markdown("""
    ### Defining Action Risk Tiers
    * **Tier 1 (Safe - Autonomous Execution)**:
      * Reading public data, calculating statistics, running localized math code, searching documentation.
    * **Tier 2 (Moderate - Alert & Proceed)**:
      * Modifying non-critical records, sending internal Slack messages, creating Jira tickets.
    * **Tier 3 (Critical - Mandatory HITL Pause)**:
      * Deleting database rows, transferring money, emailing external clients, executing destructive shell scripts, publishing public repository changes.
    """)

with col_hitl2:
    st.markdown("""
    ### State Machine Interruption Mechanics
    Modern agent frameworks (LangGraph, AutoGen) implement state persistence:
    1. Agent reaches high-risk node `execute_financial_transfer`.
    2. Graph execution **checkpoints to disk** and transitions to state `SUSPENDED_AWAITING_APPROVAL`.
    3. An approval notification is dispatched to an administrator with the exact tool parameters and rationale.
    4. Upon admin approval (via UI or Slack webhook), the graph resumes from the exact checkpoint without re-running prior reasoning steps.
    """)

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - AUDIT LOGS
# ==============================================================================
st.header("7. Audit Logs: Immutable Append-Only Traces, Cryptographic Hash Chains, and Forensic Replay", divider="rainbow")

st.markdown("""
When an autonomous agent makes decisions with legal or financial consequences, enterprises must maintain 
an **Audit Trail** capable of explaining exactly why the agent chose a specific action.
""")

st.markdown("""
### Forensic Requirements for Agent Audit Logs
1. **Full Trajectory Reconstruction**: Every log entry must capture the exact prompt, system instructions, intermediate ReAct thoughts, proposed tool calls, raw tool observations, and final responses.
2. **Cryptographic Immutability**: Implementing hash-chaining (similar to a blockchain or Git commit graph):
   `Block_Hash = SHA256(Previous_Block_Hash + Timestamp + Action_Payload + Operator_ID)`
   Any unauthorized modification of historical logs invalidates all subsequent hash signatures.
3. **Correlation ID Propagation**: A global `trace_id` passed through every downstream API call, database query, and log event to enable distributed tracing in Jaeger / OpenTelemetry.
""")

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION CODE IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Hardened Autonomous Agent Execution Gateway", divider="rainbow")

st.markdown("""
The production class below demonstrates a secure agent execution gateway featuring 
AST code inspection, risk classification, and Human-in-the-Loop approval interception.
""")

class SecureAgentGateway:
    """
    Enterprise Agent Execution Gateway implementing AST safety validation, 
    risk tiering, and human-in-the-loop approval interception.
    """
    def __init__(self):
        self.banned_ast_modules = ["os", "sys", "subprocess", "socket", "shutil", "ctypes"]
        self.banned_ast_calls = ["eval", "exec", "compile", "__import__"]
        self.high_risk_tools = ["delete_records", "send_external_email", "execute_sql_write", "transfer_funds"]

    def validate_code_ast(self, python_code: str) -> Tuple[bool, List[str]]:
        """Parses Python code into AST to detect prohibited modules and functions."""
        import ast
        violations = []
        try:
            tree = ast.parse(python_code)
        except SyntaxError as e:
            return False, [f"Syntax Error: {e.msg}"]

        for node in ast.walk(tree):
            # Check module imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.banned_ast_modules:
                        violations.append(f"Prohibited Module Import: '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                if node.module in self.banned_ast_modules:
                    violations.append(f"Prohibited Module Import: '{node.module}'")
            # Check prohibited function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in self.banned_ast_calls:
                    violations.append(f"Prohibited Builtin Call: '{node.func.id}()'")

        return len(violations) == 0, violations

    def evaluate_tool_risk(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Classifies tool execution risk into Low vs High (requiring HITL approval)."""
        is_high_risk = tool_name in self.high_risk_tools
        requires_approval = is_high_risk or ("amount" in args and args["amount"] > 1000)
        
        return {
            "tool": tool_name,
            "risk_level": "HIGH" if requires_approval else "LOW",
            "requires_hitl": requires_approval,
            "rationale": "Destructive or financial action requires operator signature" if requires_approval else "Read-only / localized action permitted"
        }

st.code('''
class SecureAgentGateway:
    def validate_code_ast(self, python_code: str) -> Tuple[bool, List[str]]:
        # Statically inspects AST tree for banned modules (os, subprocess, socket)
        ...

    def evaluate_tool_risk(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        # Classifies action risk and intercepts destructive actions for human approval
        ...
''', language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Agent Sandboxing & Tool Permission Defense Simulator", divider="rainbow")

st.markdown("""
Test how proposed agent actions and code snippets are evaluated by the **AST Code Inspector** 
and **Human-in-the-Loop Risk Gate**.
""")

agent_gate = SecureAgentGateway()

wb_col1, wb_col2 = st.columns([1.1, 0.9])

with wb_col1:
    st.subheader("1. Proposed Agent Action Simulator")
    tab_code, tab_tool = st.tabs(["Code Execution Sandbox", "API Tool Invocation"])

    with tab_code:
        code_presets = [
            "import math\nprint(f'Calculated result: {math.sqrt(144)}')",
            "import os\nos.system('rm -rf /production/data')",
            "import socket\ns = socket.socket()\ns.connect(('attacker.com', 80))",
            "data = [1, 2, 3, 4, 5]\nprint(f'Mean: {sum(data)/len(data)}')"
        ]
        selected_code = st.selectbox("Select Code Example:", code_presets)
        code_input = st.text_area("Python Script Payload:", value=selected_code, height=120)
        eval_code_btn = st.button("🛡️ Run AST Sandbox Analysis", type="primary")

    with tab_tool:
        tool_options = ["read_documentation", "query_database_select", "delete_records", "transfer_funds"]
        selected_tool = st.selectbox("Proposed Tool Call:", tool_options)
        param_value = st.number_input("Action Parameter Value / Amount ($):", min_value=10, max_value=5000, value=250)
        eval_tool_btn = st.button("⚖️ Evaluate Tool Risk Level")

with wb_col2:
    st.subheader("2. Security Perimeter Telemetry")
    if eval_code_btn or (code_input and not eval_tool_btn):
        ast_safe, ast_violations = agent_gate.validate_code_ast(code_input)
        
        m1, m2 = st.columns(2)
        with m1:
            st.metric("AST Analysis", "PASSED" if ast_safe else "REJECTED", delta="Safe" if ast_safe else "Prohibited", delta_color="normal" if ast_safe else "inverse")
        with m2:
            st.metric("Violations Found", len(ast_violations))

        if ast_safe:
            st.success("AST static inspection passed. No banned modules (os, socket, subprocess) detected.")
            st.info("Status: Dispatched to gVisor ephemeral microVM sandbox for execution.")
        else:
            st.error("SECURITY INTERCEPTION: Code rejected prior to execution!")
            for v in ast_violations:
                st.write(f"• {v}")

    elif eval_tool_btn:
        risk_result = agent_gate.evaluate_tool_risk(selected_tool, {"amount": param_value})
        
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Action Risk Tier", risk_result["risk_level"], delta="Critical" if risk_result["requires_hitl"] else "Standard", delta_color="inverse" if risk_result["requires_hitl"] else "normal")
        with m2:
            st.metric("Human Approval", "REQUIRED" if risk_result["requires_hitl"] else "AUTO-PROCEED")

        st.markdown(f"**Policy Rationale:** {risk_result['rationale']}")
        if risk_result["requires_hitl"]:
            st.warning("Action suspended in state: `AWAITING_OPERATOR_SIGNATURE`. Operator notification dispatched via webhook.")
        else:
            st.success("Action authorized. Executed automatically under standard capability scope.")

st.divider()

# ==============================================================================
# SECTION 10: INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Engineering Interview Questions")

with st.expander("Q1: Why is running agent-generated Python code inside a standard Docker container considered insecure for multi-tenant SaaS?"):
    st.markdown("""
    **Answer**: Standard Docker containers share the host machine's Linux kernel. 
    Isolation is enforced via software abstractions (namespaces and cgroups). 
    If the agent generates code that exploits a known Linux kernel vulnerability (Dirty COW, CVE-2022-0847 Dirty Pipe), 
    the process can achieve **container breakout**, gaining root access to the underlying host. 
    Production systems isolate code execution using **microVMs (Firecracker)** or **userspace kernel sandboxes (gVisor)**, 
    ensuring that hostile syscalls never directly touch the physical host kernel.
    """)

with st.expander("Q2: How does Static AST Analysis prevent Python sandbox escape attacks?"):
    st.markdown("""
    **Answer**: Attackers frequently bypass naive string matching by reconstructing banned functions dynamically:
    `getattr(__builtins__, '__imp' + 'ort__')('o' + 's').system('id')`
    or accessing object hierarchies:
    `().__class__.__base__.__subclasses__()[137]`
    **AST (Abstract Syntax Tree) Analysis** parses the code into formal grammar trees before execution, allowing security rules to inspect 
    every function call node, attribute lookup node, and import node regardless of string concatenation tricks, stopping exploits before bytecode compilation.
    """)

with st.expander("Q3: Explain the Human-in-the-Loop (HITL) checkpointing pattern in stateful agent frameworks like LangGraph."):
    st.markdown("""
    **Answer**: When an agent decides to execute a high-risk tool (e.g., executing a financial transaction or deleting records):
    1. The agent framework checkpoints its entire execution state (message history, scratchpad, pending tool parameters) to a durable database (PostgreSQL or Redis).
    2. The state transitions to `INTERRUPTED_AWAITING_APPROVAL`.
    3. The execution thread halts without holding open database locks or server memory.
    4. An external human operator reviews the proposed action and submits an approval token.
    5. The state machine reloads the checkpoint and resumes execution at the exact node without re-running earlier LLM reasoning tokens.
    """)

with st.expander("Q4: What is Server-Side Request Forgery (SSRF) in the context of autonomous web-browsing agents?"):
    st.markdown("""
    **Answer**: Autonomous agents with web-browsing or HTTP-fetch tools are instructed to retrieve URLs. 
    An attacker can inject a prompt directing the agent to fetch internal cloud metadata endpoints:
    `http://169.254.169.254/latest/meta-data/iam/security-credentials/`
    If the agent's network stack is unconstrained, the agent fetches the internal AWS IAM credentials and outputs them to the user. 
    Defenses include strict network air-gapping, blocking access to private RFC 1918 subnets, and enforcing strict domain allowlists.
    """)

with st.expander("Q5: How do Immutable Audit Logs assist in post-incident forensics for rogue agent actions?"):
    st.markdown("""
    **Answer**: In the event that an autonomous agent causes production disruption or executes unauthorized actions, 
    an immutable cryptographic audit trail (using append-only hash chains) allows security engineers to reconstruct the exact ReAct trajectory: 
    revealing whether the root cause was a malicious indirect prompt injection from a retrieved document, a model hallucination, or an operator misconfiguration.
    """)

st.subheader("Critical Production Traps")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("""
    **Trap 1: Passing Master API Keys to Agent Prompt Templates**
    * *Error*: Including static credentials (`OPENAI_API_KEY`, `STRIPE_KEY`) in system prompt strings.
    * *Impact*: Prompt injection attacks easily extract and exfiltrate production keys.
    * *Fix*: Always use backend API gateways with ephemeral vault token injection.
    """)
    st.error("""
    **Trap 2: Permitting Unbounded Execution Timeouts in Code Sandboxes**
    * *Error*: Running agent Python code without wall-clock timeouts or memory limits.
    * *Impact*: Adversarial infinite loops or memory bombs (`[0] * 10**10`) crash backend servers.
    * *Fix*: Enforce hard cgroup resource limits and 5-second SIGKILL timeouts.
    """)

with trap_c2:
    st.warning("""
    **Trap 3: Relying on String Regex for Code Safety**
    * *Warning*: Checking for `'import os'` using simple string search.
    * *Impact*: Trivially bypassed via dynamic imports (`__import__('o'+'s')`).
    * *Fix*: Mandate static AST grammar validation and run code inside a gVisor/Firecracker sandbox.
    """)
    st.warning("""
    **Trap 4: Omitting Outbound Network Blocks in Sandboxes**
    * *Warning*: Allowing sandboxed Python code to access the public internet.
    * *Impact*: Stolen corporate data is exfiltrated via DNS tunneling or HTTP POST requests.
    * *Fix*: Enforce `--net=none` to completely air-gap code execution microVMs.
    """)

st.subheader("Concept Flashcards")
fc1, fc2 = st.columns(2)
with fc1:
    with st.expander("🃏 Flashcard: Least Privilege Tooling"):
        st.markdown("**Principle**: Granting agents the minimal permissions and scopes needed for the current step.\\n\\n**Benefit**: Prevents a compromised agent from deleting data when it only needed to read data.")
with fc2:
    with st.expander("🃏 Flashcard: MicroVM vs Container"):
        st.markdown("**Docker**: Shares host Linux kernel; faster but vulnerable to kernel CVE breakouts.\\n\\n**Firecracker MicroVM**: Hardware-isolated KVM hypervisor boundary; maximum security for arbitrary code execution.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of autonomous AI agent security and sandboxing architecture:")

chk1 = st.checkbox("I understand the operational hazards of equipping probabilistic LLMs with real-world tool execution capabilities.", value=True)
chk2 = st.checkbox("I can implement Least Privilege Tool Authorization using scoped capability tokens and strict JSON Schemas.", value=True)
chk3 = st.checkbox("I can explain why microVMs (Firecracker) and gVisor sandboxes are required over standard Docker containers.", value=True)
chk4 = st.checkbox("I know how to write AST validation passes to detect and block dangerous Python modules and function calls.", value=True)
chk5 = st.checkbox("I can design a Human-in-the-Loop (HITL) state machine that checkpoints and pauses on high-risk tool proposals.", value=True)
chk6 = st.checkbox("I understand how to maintain cryptographically immutable audit logs for post-incident forensic replay.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Milestone Achieved: You have mastered Agent Security Architecture! Proceed to Lesson 22.5: Responsible AI.")
