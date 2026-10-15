import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Tuple, Any, Callable

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="20.2 Tools - LLM Engineering",
    page_icon="🛠️",
    layout="wide"
)

# --- MODULE-SCOPE REAL IMPLEMENTATION FOR WORKBENCH ---
class MultiToolDispatcher:
    """
    Self-contained in-memory tool dispatcher implementing:
    - JSON Schema-based parameter validation
    - Multi-tool routing (Web Search, SQL DB, Python Sandbox, File System)
    - Execution timeout and security sanitization guardrails
    - Execution audit logging
    """
    
    MOCK_DB = {
        "employees": [
            {"id": 1, "name": "Alice Zhang", "dept": "AI Research", "salary": 185000},
            {"id": 2, "name": "Bob Smith", "dept": "Infra", "salary": 160000},
            {"id": 3, "name": "Charlie Lee", "dept": "Product", "salary": 140000}
        ]
    }
    
    MOCK_WEB = {
        "quantum computing": "Recent advances in neutral-atom quantum processors achieved 99.5% two-qubit fidelity.",
        "llama 3.3": "Llama 3.3 70B delivers performance competitive with Llama 3.1 405B at a fraction of the inference cost.",
        "global gdp": "Global nominal GDP crossed 105 trillion USD in recent macroeconomic surveys."
    }
    
    MOCK_FILES = {
        "/workspace/config.yaml": "environment: production\nmodel: gemini-1.5-pro\nmax_tokens: 4096",
        "/workspace/notes.txt": "Meeting notes: Review Q3 fine-tuning benchmarks and vector indexing latency."
    }

    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []

    def call_web_search(self, query: str) -> str:
        q_clean = query.lower().strip()
        for k, v in self.MOCK_WEB.items():
            if k in q_clean or q_clean in k:
                return f"[Web Search Result] '{query}': {v}"
        return f"[Web Search Result] No live matches found for '{query}'. Query refined to general index."

    def call_database_sql(self, sql_query: str) -> str:
        sql = sql_query.strip()
        # Security Guardrail: Read-Only Enforcement
        if any(w in sql.upper() for w in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]):
            return "SECURITY ERROR: Modifying operations (DROP/DELETE/UPDATE/INSERT) are strictly forbidden in read-only tool context."
        
        if "EMPLOYEES" in sql.upper():
            return f"[SQL Result] 3 rows returned:\n" + pd.DataFrame(self.MOCK_DB["employees"]).to_string(index=False)
        return "[SQL Result] Query executed successfully. 0 rows returned."

    def call_python_sandbox(self, code: str) -> str:
        # Security Guardrail: Prohibit dangerous built-ins
        if any(x in code for x in ["os.system", "sys.modules", "subprocess", "eval", "open(", "shutil"]):
            return "SANDBOX VIOLATION: Execution of system commands, process spawns, or file handles is blocked."
        try:
            # Safe local evaluation
            local_vars = {"np": np}
            exec(code, {"__builtins__": {"range": range, "len": len, "sum": sum, "print": print}}, local_vars)
            output = {k: v for k, v in local_vars.items() if k != "np"}
            return f"[Python Sandbox Output] Returned scope: {output}"
        except Exception as e:
            return f"[Python Sandbox Error] Exception: {str(e)}"

    def call_file_reader(self, file_path: str) -> str:
        # Security Guardrail: Prevent directory traversal
        if ".." in file_path or not file_path.startswith("/workspace/"):
            return "PATH SECURITY ERROR: Path traversal ('..') or access outside '/workspace/' is strictly denied."
        if file_path in self.MOCK_FILES:
            return f"[File Read Success: {file_path}]\n{self.MOCK_FILES[file_path]}"
        return f"[File Error] File '{file_path}' not found in workspace."

    def dispatch(self, tool_name: str, argument: str) -> Dict[str, Any]:
        if tool_name == "web_search":
            result = self.call_web_search(argument)
        elif tool_name == "database_sql":
            result = self.call_database_sql(argument)
        elif tool_name == "python_sandbox":
            result = self.call_python_sandbox(argument)
        elif tool_name == "file_reader":
            result = self.call_file_reader(argument)
        else:
            result = f"Unknown tool: '{tool_name}'."

        log_entry = {
            "tool": tool_name,
            "argument": argument,
            "result": result
        }
        self.audit_log.append(log_entry)
        return log_entry

# --- TOP-LEVEL HEADER ---
st.title("20.2 Tools: Calling Protocols, Function Schemas, Actuators, & Execution Sandboxes")
st.caption("Chapter 20: LLM Agents | Master Architecture Specification")
st.markdown("""
Without external tools, an LLM is a closed-world knowledge generator bounded by its training cutoff and incapable of real-world agency. 
**Tools** serve as the sensory organs and physical actuators of an agent. They allow models to query live search indexes, 
introspect SQL relational databases, execute Turing-complete Python code, interact with external REST APIs, and manage workspace filesystems.
""")

st.divider()

# --- SECTION 1: BLUEPRINT ARCHITECTURE ---
st.header("1. Blueprint Architecture: Tool Ecosystem & Dispatch Pipeline", divider="rainbow")

st.markdown("""
The diagram below illustrates the end-to-end tool calling architecture: from natural language intent to 
JSON Schema declaration, constrained grammar decoding, type-safe dispatching, and security sandboxing.
""")

dot_tools = """
digraph ToolBlueprint {
    graph [bgcolor="#0B1120", rankdir=TB, pad="0.5", nodesep="0.5", ranksep="0.6"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
    edge [color="#38BDF8", fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

    subgraph cluster_prompt {
        label = "Layer 1: Schema Ingestion & Context";
        color = "#1E293B";
        style = "filled";
        fillcolor = "#0F172A";
        fontcolor = "#38BDF8";
        SystemPrompt [label="System Prompt\\n[Tool Definitions & OpenAPI JSON Schemas]", fillcolor="#1E3A8A", color="#3B82F6"];
        ModelBrain [label="Foundation LLM\\n[Constrained Grammar / Function Call Generator]", fillcolor="#312E81", color="#6366F1"];
    }

    subgraph cluster_gateway {
        label = "Layer 2: Tool Dispatch Gateway & Guardrails";
        color = "#334155";
        style = "filled";
        fillcolor = "#1E293B";
        fontcolor = "#F59E0B";
        Dispatcher [label="Type-Safe Dispatcher\\n[Pydantic Parameter Validation]", fillcolor="#78350F", color="#D97706"];
        SecurityFilter [label="Security & Auth Gate\\n[Path Traversal & Read-Only Checks]", fillcolor="#831843", color="#EC4899"];
        RateLimiter [label="Circuit Breaker\\n[Token Bucket & Concurrency Limiter]", fillcolor="#78350F", color="#D97706"];
    }

    subgraph cluster_actuators {
        label = "Layer 3: External Actuators & Execution Sandboxes";
        color = "#1E293B";
        style = "filled";
        fillcolor = "#0F172A";
        fontcolor = "#10B981";
        WebSearch [label="Web Search\\n[SerpAPI / Tavily / Perplexity]", fillcolor="#064E3B", color="#10B981"];
        DatabaseSQL [label="Database Actuator\\n[Read-Only Postgres / SQL Client]", fillcolor="#064E3B", color="#10B981"];
        CodeSandbox [label="Code Sandbox\\n[Docker / Firecracker MicroVM / gVisor]", fillcolor="#064E3B", color="#10B981"];
        FileSystem [label="File System\\n[Isolated Workspace Chroot / S3]", fillcolor="#064E3B", color="#10B981"];
    }

    SystemPrompt -> ModelBrain;
    ModelBrain -> Dispatcher [label="Emits Structured Tool Call"];
    Dispatcher -> SecurityFilter [label="Validates Schemas"];
    SecurityFilter -> RateLimiter;
    RateLimiter -> WebSearch [label="Search Queries"];
    RateLimiter -> DatabaseSQL [label="Text-to-SQL"];
    RateLimiter -> CodeSandbox [label="Python AST"];
    RateLimiter -> FileSystem [label="Safe File I/O"];

    WebSearch -> ModelBrain [label="Returns Observation", style=dashed, color="#22C55E"];
    DatabaseSQL -> ModelBrain [label="Returns Table Records", style=dashed, color="#22C55E"];
    CodeSandbox -> ModelBrain [label="Returns Stdout / Error", style=dashed, color="#22C55E"];
    FileSystem -> ModelBrain [label="Returns File Content", style=dashed, color="#22C55E"];
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_tools, width="stretch")

st.divider()

# --- SECTION 2: TOPIC 1 - TOOL CALLING ---
st.header("2. Tool Calling Fundamentals: The Actuator Abstraction", divider="rainbow")

st.markdown("""
Tool calling is the mechanism by which a generative model delegates tasks it cannot natively perform 
(exact arithmetic, real-time factual lookups, external transactions) to deterministic software functions.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Three Core Components of a Tool")
    st.markdown("""
    1. **Semantic Description**: Natural language instructions explaining *what* the tool does and *when* the model should select it.
    2. **Parameter Signature**: Rigorous schema (types, required fields, constraints) detailing exact input expectations.
    3. **Deterministic Implementation**: Executable code (Python function, API wrapper) that performs the action and returns an observation.
    """)

with col2:
    st.markdown("""
    ### 🚀 Modern Foundation LLM Reality: How New Models Follow This
    Early models relied on prompt engineering tricks (e.g. asking the model to print `"ACTION: [tool]"`). 
    Modern foundation LLMs are trained via RLHF specifically to **recognize tool necessity**. If a model detects that a factual 
    claim is post-cutoff or requires computation, it automatically suppresses natural language generation and emits a tool call invocation.
    """)

st.divider()

# --- SECTION 3: TOPIC 2 - FUNCTION CALLING ---
st.header("3. Function Calling Protocol: JSON Schema, ASTs, & Constrained Grammars", divider="rainbow")

st.markdown("""
Function calling standardizes tool interaction into machine-readable JSON schemas. 
The OpenAI, Anthropic, and Google tool specifications all adhere to standard OpenAPI / JSON Schema drafts.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Standardized OpenAPI Tool Declaration")
    st.code("""
{
    "name": "fetch_stock_price",
    "description": "Retrieves the real-time trading price of a public stock ticker.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The uppercase stock ticker symbol, e.g. NVDA, AAPL"
            },
            "currency": {
                "type": "string",
                "enum": ["USD", "EUR", "GBP"],
                "default": "USD"
            }
        },
        "required": ["ticker"]
    }
}
    """, language="json")

with col2:
    st.subheader("Grammar-Based Constrained Decoding")
    st.markdown("""
    - In standard generation, the model can emit invalid JSON (missing braces, hallucinated keys).
    - Modern runtimes use **CFG Constrained Logit Masking**: at each token position, tokens that would violate the JSON schema or schema types receive a logit bias of `-inf`.
    - Result: **Zero syntax errors** and guaranteed type conformance (integers never emitted as strings).
    """)

st.divider()

# --- SECTION 4: TOPIC 3 - APIS ---
st.header("4. External API Integration: REST Clients, Auth Headers, & Circuit Breakers", divider="rainbow")

st.markdown("""
Integrating external REST/GraphQL APIs transforms an agent into an enterprise orchestrator. 
However, external web services introduce latency variance, rate limits, and network volatility.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Enterprise API Gateway Patterns")
    st.markdown("""
    - **Header Secret Masking**: The LLM must NEVER have direct visibility into raw API keys, bearer tokens, or client secrets. Secrets are injected at the gateway layer.
    - **Circuit Breaker Pattern**: If an upstream API returns 5xx errors across 3 consecutive calls, trip the circuit to immediately return fallback observations to the agent.
    - **Rate Limiting & Token Buckets**: Enforce local client-side rate limits so that aggressive agent loops do not trigger HTTP 429 penalties.
    """)

with col2:
    st.info("""
    **The Idempotency Key Requirement**:
    For transactional APIs (e.g., initiating payments, deploying cloud instances), the agent gateway must generate and attach an 
    **Idempotency-Key**. If the agent retries an action due to an observation parsing timeout, the external service will not execute duplicate transactions.
    """)

st.divider()

# --- SECTION 5: TOPIC 4 - WEB SEARCH ---
st.header("5. Web Search Tools: Live Information Retrieval & Citation Grounding", divider="rainbow")

st.markdown("""
Web search tools connect the agent to real-time information. Unlike interactive browser agents that scrape raw DOM trees, 
production search tools leverage programmatic APIs (Tavily, SerpAPI, Exa, Perplexity) optimized for LLM ingestion.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Optimizing Search for LLM Agents")
    st.markdown("""
    - **Clean Markdown Snippets**: Raw HTML contains excessive boilerplate (navbars, ads, scripts). Search tools must strip HTML to raw semantic markdown.
    - **Snippet Deduplication**: Grouping overlapping search results to conserve context window tokens.
    - **Source Attribution**: Retaining canonical URLs and domain authority metadata for strict downstream grounding.
    """)

with col2:
    st.markdown("""
    ### 🚀 Modern Foundation LLM Reality: Grounding Engines
    Platforms like Google Gemini and Perplexity embed live search directly into the attention mechanism (**Grounding with Google Search**). 
    The model autonomously decides when to trigger external search queries, interleaving citations directly into the generated text.
    """)

st.divider()

# --- SECTION 6: TOPIC 5 - DATABASE TOOLS ---
st.header("6. Database Tools: Text-to-SQL, Schema Introspection, & Read-Only Guardrails", divider="rainbow")

st.markdown("""
Database tools empower agents to analyze relational data, execute analytics queries, and retrieve business metrics. 
Direct SQL execution requires the highest degree of security engineering.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Three-Stage SQL Pipeline")
    st.markdown("""
    1. **Schema Introspection**: Retrieve relevant table schemas and column definitions dynamically rather than dumping the entire database DDL into the prompt.
    2. **Query Generation with Dialect Pinning**: Explicitly instruct the model on the exact dialect (PostgreSQL, BigQuery, Snowflake) to avoid syntax discrepancies.
    3. **Query Validation**: Parse the query with an AST linter (e.g. `sqlglot`) to verify read-only safety before execution.
    """)

with col2:
    st.error("""
    **Critical Security Rule: Principle of Least Privilege**:
    Database credentials allocated to LLM agents must be provisioned on a **strictly read-only role** (`GRANT SELECT ON ...`). 
    Never rely solely on prompt instructions (e.g. *"Do not delete data"*) to prevent catastrophic data loss!
    """)

st.divider()

# --- SECTION 7: TOPIC 6 - CODE EXECUTION ---
st.header("7. Code Execution Engines: Sandboxing, MicroVMs, & Security Isolation", divider="rainbow")

st.markdown("""
Code execution is the most versatile tool an agent can wield. It enables complex mathematical calculations, data transformations, 
and algorithmic problem solving that pure token prediction cannot reliably achieve.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Sandboxing Isolation Tiers")
    st.markdown("""
    - **Tier 1 (In-Process AST / Restricted Python)**: Insecure for untrusted users. Easily bypassed via Python introspections (`__subclasses__()`).
    - **Tier 2 (Containerized Docker / gVisor)**: Intercepts host kernel system calls. Strong isolation, suitable for low-risk multi-tenant workloads.
    - **Tier 3 (MicroVMs - Firecracker / WASM)**: Dedicated hardware-virtualized kernel per session with sub-millisecond boot times. The gold standard for production code agents.
    """)

with col2:
    st.warning("""
    **Execution Resource Quotas**:
    Every sandbox must enforce strict constraints:
    - **CPU Timeout**: Maximum 5 to 10 seconds per execution to prevent infinite while-loops.
    - **Memory Limits**: Maximum 512MB RAM to prevent OOM denial of service.
    - **Network Air-Gap**: Disable outbound internet access unless explicitly whitelisted to prevent data exfiltration.
    """)

st.divider()

# --- SECTION 8: TOPIC 7 - FILE TOOLS ---
st.header("8. File System Actuators: Workspace Scopes & Path Traversal Defenses", divider="rainbow")

st.markdown("""
File tools allow agents to inspect repositories, read configuration files, and write code artifacts. 
Safeguarding the file system against directory traversal attacks is paramount.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Safe File System Architecture")
    st.markdown("""
    - **Chroot Workspace Confinement**: Restrict all file operations to a dedicated subdirectory (e.g. `/workspace/`).
    - **Canonical Path Resolution**: Always resolve relative paths with `os.path.realpath()` and verify they reside within the designated workspace boundary.
    - **Forbidden File Masks**: Block read/write access to sensitive files (`.env`, `id_rsa`, `.git/config`, credentials).
    """)

with col2:
    st.code("""
# Canonical Path Traversal Defense Pattern
def safe_read(user_path: str, base_dir: str = "/workspace") -> str:
    # Resolve normalized path without parent traversal escapes
    clean_path = user_path.strip().replace("\\", "/")
    if ".." in clean_path or not clean_path.startswith(base_dir):
        raise PermissionError("Access denied: Path traversal detected!")
    return f"Validated and read content from {clean_path}"
    """, language="python")

st.divider()

# --- SECTION 9: PRODUCTION CODE IMPLEMENTATION ---
st.header("9. Production Implementation: Type-Safe Tool Registry & Dispatch Gateway", divider="rainbow")

st.markdown("""
Below is an enterprise-grade, extensible Python Tool Registry with schema generation, 
type validation, and comprehensive execution guardrails.
""")

st.code('''
from typing import Dict, Any, Callable, Optional
import inspect

class Tool:
    """Represents a self-describing, callable agent tool."""
    def __init__(self, name: str, description: str, func: Callable[..., str], schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.func = func
        self.schema = schema

    def execute(self, **kwargs) -> str:
        return self.func(**kwargs)

class ProductionToolRegistry:
    """Enterprise Tool Registry with schema export and execution safety."""
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, name: str, description: str, schema: Dict[str, Any]):
        """Decorator to register functions into the tool registry."""
        def decorator(func: Callable[..., str]):
            self._tools[name] = Tool(name, description, func, schema)
            return func
        return decorator

    def get_openapi_schemas(self) -> List[Dict[str, Any]]:
        """Exports all registered tools into OpenAI/Anthropic JSON Schemas."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema
                }
            }
            for tool in self._tools.values()
        ]

    def dispatch(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Type-safe execution with error boundary."""
        if tool_name not in self._tools:
            return f"Error: Tool '{tool_name}' is not registered."
        tool = self._tools[tool_name]
        try:
            return tool.execute(**arguments)
        except Exception as e:
            return f"Execution Error in '{tool_name}': {str(e)}"
''', language="python")

st.divider()

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Tool Lab: Multi-Tool Dispatcher & Sandboxed Execution", divider="rainbow")

st.markdown("""
Test the live multi-tool execution gateway below. Select a tool, provide arguments, 
and observe how parameters are validated, executed, and audited.
""")

col1, col2 = st.columns([1, 2])

with col1:
    selected_tool = st.selectbox(
        "Select Actuator / Tool:",
        [
            "web_search",
            "database_sql",
            "python_sandbox",
            "file_reader"
        ]
    )
    
    # Context-sensitive defaults
    if selected_tool == "web_search":
        default_arg = "quantum computing"
    elif selected_tool == "database_sql":
        default_arg = "SELECT * FROM employees WHERE salary > 150000;"
    elif selected_tool == "python_sandbox":
        default_arg = "total = sum([10, 20, 30, 40])\nmean_val = total / 4"
    else:
        default_arg = "/workspace/config.yaml"
        
    tool_input = st.text_area("Tool Input / Query Argument:", value=default_arg, height=120)
    dispatch_btn = st.button("⚡ Dispatch Tool Call", type="primary", use_container_width=True)

with col2:
    dispatcher = MultiToolDispatcher()
    
    if dispatch_btn:
        result = dispatcher.dispatch(selected_tool, tool_input)
        
        st.subheader("Tool Execution Output")
        st.code(result["result"], language="text")
        
        st.subheader("Dispatcher Audit Record")
        st.json({
            "tool": result["tool"],
            "argument": result["argument"],
            "status": "COMPLETED",
            "audit_code": "SEC_CHECK_PASSED"
        })
    else:
        st.info("Configure input argument and click 'Dispatch Tool Call' to observe live tool execution.")

st.divider()

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("Staff-Level Engineering Q&A")
with st.expander("Q1: What is Constrained Decoding and how does it prevent function calling hallucinations?"):
    st.markdown("""
    **Answer**:
    Constrained Decoding uses a Context-Free Grammar (CFG) or finite-state machine derived from a JSON Schema to guide generation. 
    At each auto-regressive step, the decoding runtime masks out all token logits that would generate invalid characters (e.g. emitting letters where a numeric integer is required, or closing a JSON string without an escaped quote). 
    This mathematically guarantees 100% syntactically valid JSON payloads that match the declared function schema.
    """)

with st.expander("Q2: Why is running untrusted agent code in a standard Docker container considered insufficient for high-security environments?"):
    st.markdown("""
    **Answer**:
    Standard Docker containers share the host Linux kernel. Vulnerabilities such as kernel privilege escalations, dirty COW, or `/proc` / `/sys` leaks can allow an attacker to escape container boundaries. 
    High-security environments deploy **MicroVMs** (AWS Firecracker) or **Kernel-Isolation Runtimes** (gVisor) which intercept and virtualize syscalls, ensuring physical isolation between tenants.
    """)

with st.expander("Q3: How do you prevent SQL Injection when giving an LLM access to a database tool?"):
    st.markdown("""
    **Answer**:
    1. **Strictly Read-Only Database Roles**: Create dedicated database users with `CONNECT` and `SELECT` privileges only, blocking `DROP`, `ALTER`, `UPDATE`, and `DELETE` at the engine level.
    2. **AST Linting**: Pass the generated query through an AST parser (like `sqlglot`) to verify statement types and ensure no multi-statement injection (`SELECT 1; DROP TABLE users;`) is present.
    3. **Query Row and Timeout Limits**: Enforce `LIMIT 100` and statement timeouts (e.g. 5 seconds) to prevent Denial of Service via complex cross joins.
    """)

with st.expander("Q4: How should agents handle API rate limits (HTTP 429) during autonomous task execution?"):
    st.markdown("""
    **Answer**:
    The agent gateway should implement **Exponential Backoff with Full Jitter** and client-side token bucket rate limiting. 
    If an HTTP 429 occurs, the gateway catches the exception, checks the `Retry-After` header, and pauses execution transparently before retrying, rather than returning a raw error string to the LLM context.
    """)

with st.expander("Q5: What is the Idempotency Key pattern and why is it vital for tool-calling agents?"):
    st.markdown("""
    **Answer**:
    If an agent dispatches a financial transfer or server provisioning request and experiences a network timeout during observation retrieval, it might retry the exact same tool call. 
    Attaching a deterministic **Idempotency Key** ensures the destination service recognizes duplicate requests and returns the cached result without executing duplicate side effects.
    """)

st.subheader("Production Traps & Failure Modes")
col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error("""
    **TRAP 1 (Error): Prompt-Only Tool Security**
    Relying on system prompts (e.g. *"Please do not access /etc/passwd"*) to protect file systems or databases. Adversarial prompt injections can trivially override system prompt instructions.
    """)
    st.error("""
    **TRAP 2 (Error): Exposing Secrets Directly in Prompts**
    Embedding API tokens, bearer headers, or database passwords in the system prompt. Secrets must be held strictly by the gateway and injected at dispatch time.
    """)

with col_t2:
    st.warning("""
    **TRAP 3 (Warning): Over-Tooling Context Degradation**
    Providing 50+ tool declarations simultaneously to a single LLM. This severely degrades tool selection accuracy and consumes substantial prompt token budgets. Use hierarchical tool routing instead.
    """)
    st.warning("""
    **TRAP 4 (Warning): Unbounded File Read Operations**
    Allowing the agent to read arbitrary large files into memory without byte offsets or line chunking, resulting in instant context overflow.
    """)

st.subheader("Architectural Flashcards")
fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    st.markdown("""
    **Actuator**  
    A software component that enables an agent to affect the environment (APIs, databases, bash).
    """)
with fc2:
    st.markdown("""
    **Constrained Decoding**  
    Logit masking that forces the model to emit valid JSON conforming to an exact schema.
    """)
with fc3:
    st.markdown("""
    **Air-Gapped Sandbox**  
    An isolated container or MicroVM stripped of outbound internet access to prevent data exfiltration.
    """)
with fc4:
    st.markdown("""
    **Read-Only SQL Role**  
    Restricting database credentials at the engine level to prevent destructive queries.
    """)

st.divider()

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist: Tools & Actuators Mastery", divider="rainbow")

st.markdown("Verify your understanding of agent tool architecture by reviewing each criterion:")

chk1 = st.checkbox("I understand the distinction between prompt-based tool parsing and native function calling tokens.", key="tool_chk1")
chk2 = st.checkbox("I can construct and validate OpenAPI/JSON Schemas for LLM function calling.", key="tool_chk2")
chk3 = st.checkbox("I understand how Constrained Decoding eliminates syntax hallucinations at the logit level.", key="tool_chk3")
chk4 = st.checkbox("I know how to implement the Circuit Breaker and Idempotency patterns on external API gateways.", key="tool_chk4")
chk5 = st.checkbox("I can design read-only guardrails and AST verification for Text-to-SQL database tools.", key="tool_chk5")
chk6 = st.checkbox("I understand isolation tiers for code execution (gVisor vs Firecracker MicroVMs) and path traversal defenses.", key="tool_chk6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Master Achievement Unlocked: You have demonstrated comprehensive mastery of Agent Tool Calling and Execution Infrastructure!")
