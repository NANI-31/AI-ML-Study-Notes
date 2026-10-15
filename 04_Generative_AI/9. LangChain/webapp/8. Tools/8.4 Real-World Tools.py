import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="8.4 Real-World Tools - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("8.4 Real-World Tools: Search, Databases, APIs, Sandboxes & Business Systems")
st.caption("Chapter 8: Tools | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
In enterprise AI deployments, agents must interface with complex, heterogeneous infrastructure: 
web search engines, production relational databases, cloud REST APIs, sandboxed code executors, 
and internal CRM/ERP applications.

Deploying real-world tools requires mastering not only the API connections, but also **security sandboxing**, 
**read-only guardrails**, **credential isolation**, and **rate-limit management**.

In this lesson, you will master the 8 foundational categories of real-world tools:
1. **Web Search Tools**: Grounding agents with live web search APIs (Tavily, DuckDuckGo).
2. **Database Tools**: Securely querying relational databases with `SQLDatabaseToolkit` and read-only guards.
3. **Calculator Tools**: Providing algorithmic precision and zero hallucination for financial and mathematical computations.
4. **API Tools**: Interacting with external REST APIs using OpenAPI schemas and authenticated requests.
5. **File System Tools**: Safely reading and writing files while defending against directory traversal attacks.
6. **Python Execution Tools**: Running sandboxed code with resource quotas and security containment.
7. **Email & Messaging Tools**: Dispatching outbound communications with human-in-the-loop safeguards.
8. **Custom Business Tools**: Encapsulating proprietary microservices, ticketing platforms, and ERP workflows.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockToolRunReport:
    """Represents a simulated tool execution event in module scope."""
    def __init__(self, tool_category: str, input_params: Dict[str, Any], output_summary: str, security_audit: str, latency_ms: int):
        self.tool_category = tool_category
        self.input_params = input_params
        self.output_summary = output_summary
        self.security_audit = security_audit
        self.latency_ms = latency_ms

class MockRealWorldToolSimulator:
    """Simulates multi-domain enterprise tools with security auditing in module scope."""
    def __init__(self):
        pass

    def execute_tool(self, tool_type: str, query: str) -> MockToolRunReport:
        query_clean = query.strip()

        if tool_type == "Web Search (Tavily / DuckDuckGo)":
            output = f"Top Result: 'LangChain v0.3 introduces improved LCEL streaming and simplified tool calling.' (Source: https://blog.langchain.dev, Relevance: 0.96)"
            audit = "Outbound HTTPS GET to search API. User identity anonymized. Results filtered for safe search."
            latency = 310
            params = {"query": query_clean, "max_results": 3, "search_depth": "advanced"}

        elif tool_type == "SQL Database Query":
            # Security check for mutating queries
            if any(banned in query_clean.upper() for banned in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]):
                output = "SECURITY_VIOLATION: Mutating queries are rejected. Read-only SELECT transactions enforced."
                audit = "BLOCKED: Static SQL AST parser intercepted unauthorized mutation keyword."
                latency = 8
            else:
                output = "[{'region': 'us-east', 'revenue_usd': 1420500.0, 'active_tenants': 84}]"
                audit = "PASSED: Query executed inside read-only transaction with 50-row result limit."
                latency = 45
            params = {"sql": query_clean, "read_only": True}

        elif tool_type == "Calculator / Math Engine":
            try:
                # Safe eval simulation
                clean_math = "".join(c for c in query_clean if c in "0123456789+-*/.() ")
                val = eval(clean_math, {"__builtins__": None}, {})
                output = f"Exact Calculation Result: {val}"
                audit = "AST Sanitized: Only pure arithmetic operators allowed. Zero code execution risk."
            except Exception as e:
                output = f"Math Evaluation Error: {e}"
                audit = "Syntax error in mathematical expression."
            latency = 4
            params = {"expression": query_clean}

        elif tool_type == "Python Sandbox (e2b / Docker)":
            if "os." in query_clean or "subprocess" in query_clean:
                output = "SANDBOX_SECURITY_VIOLATION: Access to 'os' and 'subprocess' blocked by container seccomp profile."
                audit = "BLOCKED: Container security policy prevented dangerous syscall invocation."
                latency = 12
            else:
                output = "Standard Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\nExecution finished with exit code 0."
                audit = "PASSED: Executed inside isolated ephemeral microVM with 256MB RAM and 2.0s CPU timeout."
                latency = 420
            params = {"script": query_clean[:50] + "..."}

        else:  # Custom ERP / CRM Tool
            output = f"Salesforce Record SF-94812 updated: Account 'Acme Corp' stage moved to 'Negotiation'. Dispatched notification."
            audit = "OAuth2 Bearer Token authenticated. Audit log written to corporate SIEM cluster."
            latency = 190
            params = {"crm_action": "update_stage", "account": "Acme Corp"}

        return MockToolRunReport(
            tool_category=tool_type,
            input_params=params,
            output_summary=output,
            security_audit=audit,
            latency_ms=latency
        )

# --- SECTION 1: ARCHITECTURE BLUEPRINT & ENTERPRISE TOOL ECOSYSTEM ---
st.header("1. Architecture Blueprint: The Enterprise Tool Ecosystem & Secure Integration Layer", divider="rainbow")

st.markdown("""
Enterprise real-world tools operate across distinct trust domains. 
To protect corporate infrastructure, the **Secure Integration Layer** sits between the AI Agent 
and external systems, enforcing authentication, rate-limiting, static SQL AST parsing, and sandbox isolation.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph G {
        bgcolor="#0B1120";
        rankdir=TB;
        splines=spline;
        node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B", arrowsize=0.8, penwidth=1.2];

        // Tier 1: Agent Reasoning
        subgraph cluster_agent_tier {
            label = "1. AGENT COGNITIVE TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            AIAgent [label="AI Agent Controller (LCEL / LangGraph)\\nEmits ToolCall(name='...', args={...})"];
        }

        // Tier 2: Security & Governance Boundary
        subgraph cluster_security_tier {
            label = "2. SECURITY &amp; GOVERNANCE BARRIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#0F172A"];
            SecurityFilter [label="Security Proxy &amp; Validator\\nSanitizes input, checks RBAC &amp; credentials", color="#10B981"];
            HITLGate [label="Human-In-The-Loop Approval Gate\\nRequired for destructive actions (Email / Delete)", color="#F59E0B"];
        }

        // Tier 3: Domain Tool Categories
        subgraph cluster_tools_tier {
            label = "3. ENTERPRISE TOOL WORKERS";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            WebSearchTool [label="Web Search Engine\\nTavily / DuckDuckGo API"];
            DatabaseTool [label="SQL Database Gateway\\nRead-Only Transaction Pool", color="#38BDF8"];
            PythonSandbox [label="Sandboxed Python Runtime\\ne2b / Docker MicroVM", color="#38BDF8"];
            BusinessAPITool [label="Internal Business Services\\nJira / Salesforce / Slack REST API"];
        }

        // Tier 4: External Systems
        subgraph cluster_systems {
            label = "4. TARGET ENTERPRISE SYSTEMS";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#0F172A"];
            InternetWeb [label="Public Web &amp; News"];
            PostgresDB [label="PostgreSQL Database Cluster"];
            DockerRuntime [label="Isolated MicroVM Container"];
            EnterpriseSaaS [label="SaaS &amp; Internal Microservices"];
        }

        // Wiring
        AIAgent -> SecurityFilter;
        SecurityFilter -> HITLGate [label="Destructive / Side-effect action"];
        SecurityFilter -> WebSearchTool [label="Read-only query"];
        SecurityFilter -> DatabaseTool [label="Read-only SQL"];
        SecurityFilter -> PythonSandbox [label="Code execution"];
        
        HITLGate -> BusinessAPITool [label="Operator Approved"];

        WebSearchTool -> InternetWeb;
        DatabaseTool -> PostgresDB;
        PythonSandbox -> DockerRuntime;
        BusinessAPITool -> EnterpriseSaaS;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
**Core Security Principles for Enterprise Tools:**
1. **Principle of Least Privilege**: Database tools must connect with dedicated read-only credentials; file tools must be chrooted.
2. **Human-in-the-Loop for Side Effects**: Any tool that mutates state (sending emails, modifying customer records, deploying code) should require explicit operator confirmation before execution.
3. **Air-Gapped Code Execution**: Python execution tools must run inside isolated microVMs (e.g. e2b, Modal, or Docker) with strict memory, CPU, and network quotas.
""")

st.divider()

# --- SECTION 2: WEB SEARCH TOOLS ---
st.header("2. Web Search Tools (Tavily, DuckDuckGo & Real-Time Grounding)", divider="rainbow")

st.markdown("""
Web search tools connect AI models to the live internet. 
Unlike standard consumer search engines designed for humans, modern AI search tools (such as **Tavily**) 
are optimized specifically for LLMs: they extract clean markdown content, filter out ads and cookie banners, 
and return relevant snippets with verifiable URLs.
""")

st.code('''from langchain_community.tools.tavily_search import TavilySearchResults

# Tavily is explicitly designed for LLM agents:
search_tool = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_domains=["github.com", "arxiv.org", "python.org"]
)

# Invocation returns structured snippets with citations:
results = search_tool.invoke({"query": "What is new in Python 3.13 free threading?"})
for r in results:
    print(f"Title: {r['url']}")
    print(f"Snippet: {r['content'][:150]}...\\n")
''', language="python")

st.divider()

# --- SECTION 3: DATABASE TOOLS ---
st.header("3. Database Tools (`SQLDatabaseToolkit` & Read-Only Governance)", divider="rainbow")

st.markdown("""
Allowing an LLM to query a relational database enables autonomous data analysis and natural language reporting. 
However, exposing production databases to an LLM requires strict defensive engineering:
- **Schema Pruning**: Expose only relevant tables and columns to protect sensitive PII and conserve context window tokens.
- **Read-Only Credentials**: Use a dedicated database user with `GRANT SELECT` privileges only.
- **SQL AST Verification**: Use a static SQL parser to reject statements containing `DROP`, `DELETE`, `UPDATE`, or `ALTER`.
""")

st.code('''from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

# 1. Connect with read-only connection string
db = SQLDatabase.from_uri(
    "postgresql://analytics_readonly:password@db.prod.internal:5432/warehouse",
    include_tables=["customer_metrics", "daily_revenue"]  # Prune tables
)

model = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Toolkit provides tools for schema inspection, query generation, and execution
toolkit = SQLDatabaseToolkit(db=db, llm=model)
sql_tools = toolkit.get_tools()

# Tools included in toolkit:
# - sql_db_list_tables: List available tables in database
# - sql_db_schema: Get column schema and sample rows for given tables
# - sql_db_query: Execute a read-only query
# - sql_db_query_checker: Validate SQL syntax before execution
''', language="python")

st.divider()

# --- SECTION 4: CALCULATOR TOOLS ---
st.header("4. Calculator & Math Tools (Deterministic Computational Precision)", divider="rainbow")

st.markdown("""
LLMs frequently hallucinate multi-digit arithmetic or floating-point calculations because tokenizers 
split numbers arbitrarily across token boundaries. Calculator tools provide guaranteed mathematical precision.
""")

st.code('''from langchain_core.tools import tool
import math

@tool
def evaluate_financial_formula(formula: str) -> str:
    """Evaluate complex mathematical and financial formulas with exact precision.
    Supports standard arithmetic (+, -, *, /), powers (**), and math functions (sqrt, log).
    
    Args:
        formula: Mathematical string expression (e.g. '1000 * (1 + 0.08 / 12) ** (12 * 5)').
    """
    safe_namespace = {
        "sqrt": math.sqrt,
        "log": math.log,
        "exp": math.exp,
        "pi": math.pi,
        "round": round,
        "__builtins__": None
    }
    try:
        # Strictly sandboxed mathematical evaluation
        result = eval(formula, safe_namespace, {})
        return str(result)
    except Exception as e:
        return f"Formula evaluation error: {e}"
''', language="python")

st.divider()

# --- SECTION 5: API TOOLS ---
st.header("5. API Tools (REST Endpoints & OpenAPI Integrations)", divider="rainbow")

st.markdown("""
Most enterprise software is accessed via REST APIs. 
LangChain allows you to either create individual tools wrapping HTTP endpoints or generate a complete 
toolkit dynamically from an **OpenAPI / Swagger** specification.
""")

st.code('''import requests
from langchain_core.tools import tool
from typing import Dict, Any

@tool
def query_weather_api(latitude: float, longitude: float) -> Dict[str, Any]:
    """Fetch live weather metrics from Open-Meteo REST API.
    
    Args:
        latitude: Geographical latitude coordinate.
        longitude: Geographical longitude coordinate.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    response = requests.get(url, timeout=5.0)
    response.raise_for_status()
    data = response.json()
    return data.get("current_weather", {})
''', language="python")

st.divider()

# --- SECTION 6: FILE SYSTEM TOOLS ---
st.header("6. File System Tools (Safe File Access & Path Traversal Defenses)", divider="rainbow")

st.markdown("""
File system tools allow agents to inspect source code, ingest reports, and write temporary outputs. 
However, naive file tools are susceptible to **Directory Traversal Attacks** (e.g., an LLM requesting `../../etc/passwd`).

**Defense Standard**: Enforce a strict root sandbox directory and verify that all requested paths 
resolve within that boundary.
""")

st.code('''import os
from langchain_core.tools import tool

SANDBOX_ROOT = "/app/agent_workspace"

@tool
def read_workspace_file(relative_path: str) -> str:
    """Read contents of a file inside the sandboxed workspace.
    
    Args:
        relative_path: Relative path to target file (e.g. 'reports/summary.txt').
    """
    # Defensive path resolution to prevent directory traversal
    resolved_path = os.path.realpath(os.path.join(SANDBOX_ROOT, relative_path))
    
    if not resolved_path.startswith(os.path.realpath(SANDBOX_ROOT)):
        return "SECURITY_ERROR: Access denied. Directory traversal outside workspace root is blocked."
        
    if not os.path.exists(resolved_path):
        return f"Error: File '{relative_path}' does not exist."
        
    with open(resolved_path, "r", encoding="utf-8") as f:
        return f.read()
''', language="python")

st.divider()

# --- SECTION 7: PYTHON CODE EXECUTION TOOLS ---
st.header("7. Python Execution Tools (Sandboxed Code Execution)", divider="rainbow")

st.markdown("""
Data analysis and algorithmic problem solving often require writing and running Python scripts on the fly. 
Running un-sandboxed code generated by an LLM on your host machine is a critical security vulnerability.

Production architectures execute agent code in **isolated microVMs or sandboxes** (such as e2b, Modal, or Docker).
""")

st.code('''# Conceptual pattern using a sandboxed code execution tool:
from langchain_core.tools import tool

@tool
def execute_python_in_sandbox(code: str) -> str:
    """Execute Python data analysis code inside an isolated ephemeral microVM sandbox.
    Has access to pandas, numpy, matplotlib, and scikit-learn.
    Network access is disabled. Timeout: 10 seconds.
    
    Args:
        code: Complete executable Python code block.
    """
    # In production: dispatch code to e2b CodeInterpreter or ephemeral Docker container
    # sandbox = e2b.CodeInterpreter()
    # execution = sandbox.run_code(code)
    # return execution.text
    return "Simulated execution output: Array shape (1000, 4). Mean accuracy: 0.942."
''', language="python")

st.divider()

# --- SECTION 8: EMAIL & COMMUNICATION TOOLS ---
st.header("8. Email & Messaging Tools (Notifications & Human-in-the-Loop)", divider="rainbow")

st.markdown("""
Outbound communications (emails, Slack messages, PagerDuty incidents) have high real-world stakes. 
An autonomous agent should never send an external communication without passing through a 
**Human-in-the-Loop (HITL) gate** or dry-run preview stage.
""")

st.code('''from langchain_core.tools import tool

@tool
def draft_customer_outreach_email(recipient_email: str, subject: str, body: str) -> str:
    """Draft an outbound customer email. 
    NOTE: This creates a DRAFT in the CRM queue for human review. It does NOT send immediately.
    
    Args:
        recipient_email: Valid recipient email address.
        subject: Email subject line.
        body: Markdown formatted email content.
    """
    # Writes draft to CRM / SendGrid drafts queue
    draft_id = "DRAFT-94812"
    return f"Email successfully drafted (ID: {draft_id}). Pending human review before dispatch."
''', language="python")

st.divider()

# --- SECTION 9: CUSTOM BUSINESS TOOLS ---
st.header("9. Custom Business Tools (Proprietary ERP, CRM & Microservices)", divider="rainbow")

st.markdown("""
The true power of AI agents lies in orchestrating proprietary corporate workflows: checking inventory in SAP, 
updating opportunities in Salesforce, or opening engineering tickets in Jira.
""")

st.code('''from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from typing import Type

class JiraTicketInput(BaseModel):
    project_key: str = Field(description="Jira project key e.g. 'ENG', 'INFRA'")
    summary: str = Field(description="Concise defect summary title")
    description: str = Field(description="Full markdown reproduction steps")
    priority: str = Field(default="Medium", description="Priority: Low, Medium, High, Blocker")

class JiraIssueCreatorTool(BaseTool):
    name: str = "create_jira_issue"
    description: str = "Create a bug or engineering task in internal Jira workspace."
    args_schema: Type[BaseModel] = JiraTicketInput

    def _run(self, project_key: str, summary: str, description: str, priority: str = "Medium") -> str:
        # In production: Call Jira REST API
        ticket_key = f"{project_key}-1042"
        return f"Created Jira issue {ticket_key}: https://jira.internal.corp/browse/{ticket_key}"
''', language="python")

st.divider()

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: Enterprise Multi-Domain Tool Suite", divider="rainbow")

st.markdown("""
The following complete production pattern illustrates an **Enterprise Multi-Domain Tool Suite** 
that integrates web search, read-only SQL, sandboxed math, and internal ticketing behind 
a unified governance proxy.
""")

st.code('''import logging
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger("EnterpriseToolSuite")

# Tool 1: Real-time Search
@tool
def enterprise_search(query: str) -> str:
    """Search internal documentation and public technical articles."""
    return f"Search result for '{query}': Found verified runbook on cluster scaling."

# Tool 2: Read-Only Database
@tool
def query_analytics_store(sql_query: str) -> str:
    """Execute read-only SQL queries against the analytics warehouse."""
    if any(k in sql_query.upper() for k in ["DROP", "DELETE", "UPDATE", "INSERT"]):
        raise PermissionError("Mutating SQL queries are rejected.")
    return f"Executed '{sql_query}': [{{'active_nodes': 128, 'cpu_load': 0.64}}]"

# Tool 3: Mathematical Engine
@tool
def calculate_metric(expression: str) -> str:
    """Perform exact numerical calculations."""
    return str(eval(expression, {"__builtins__": None}, {}))

class EnterpriseAgentService:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0)
        self.tools = [enterprise_search, query_analytics_store, calculate_metric]
        self.bound_model = self.model.bind_tools(self.tools)

    def process_request(self, prompt: str) -> Dict[str, Any]:
        logger.info(f"Processing enterprise agent request: {prompt}")
        response = self.bound_model.invoke([HumanMessage(content=prompt)])
        return {
            "model_response": response.content,
            "tool_calls_requested": response.tool_calls
        }
''', language="python")

st.divider()

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: Real-World Enterprise Tool Simulator & Sandbox", divider="rainbow")

st.markdown("""
Test how different categories of enterprise tools behave under realistic inputs. 
Inspect security validation audits (such as blocking mutating SQL or catching sandbox escape attempts) in real time.
""")

tool_sim = MockRealWorldToolSimulator()

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("🛠️ Select Tool Domain & Input")
    available_tools = [
        "Web Search (Tavily / DuckDuckGo)",
        "SQL Database Query",
        "Calculator / Math Engine",
        "Python Sandbox (e2b / Docker)",
        "Custom Business Tool (Salesforce CRM)"
    ]
    selected_tool = st.selectbox("Tool Domain", available_tools)

    presets = {
        "Web Search (Tavily / DuckDuckGo)": "What are the latest security patches in Kubernetes 1.31?",
        "SQL Database Query": "SELECT region, revenue_usd FROM sales WHERE revenue_usd > 1000000;",
        "Calculator / Math Engine": "48200 * (1 + 0.045 / 12) ** (12 * 3)",
        "Python Sandbox (e2b / Docker)": "import numpy as np\nvals = np.array([10, 20, 30])\nprint(vals.mean())",
        "Custom Business Tool (Salesforce CRM)": "Update Acme Corp account stage to Negotiation"
    }

    user_query = st.text_area("Input Parameter / Query Payload", value=presets[selected_tool], height=120)

with col_wb2:
    st.subheader("🔒 Active Security Governance Rules")
    if "SQL" in selected_tool:
        st.info("**SQL Governance**: Only SELECT statements permitted. Static parser blocks mutating keywords (`DROP`, `DELETE`, `UPDATE`).")
    elif "Python" in selected_tool:
        st.info("**Sandbox Governance**: MicroVM isolation. Seccomp profile blocks dangerous syscalls (`os`, `subprocess`).")
    elif "Web Search" in selected_tool:
        st.info("**Search Governance**: Outbound HTTPS calls anonymized. PII stripped from query parameters.")
    else:
        st.info("**General Governance**: Role-based access control (RBAC) enforced with centralized SIEM audit logging.")

if st.button("🚀 Execute Tool in Sandbox", type="primary"):
    res = tool_sim.execute_tool(selected_tool, user_query)
    
    st.subheader("📊 Execution & Security Telemetry")
    m1, m2 = st.columns(2)
    m1.metric("Tool Category", res.tool_category)
    m2.metric("Execution Latency", f"{res.latency_ms} ms")

    st.write("#### 🛡️ Security Audit Log:")
    if "BLOCKED" in res.security_audit:
        st.error(res.security_audit)
    else:
        st.success(res.security_audit)

    st.write("#### 📦 Tool Observation Payload (`ToolMessage.content`):")
    if "SECURITY_VIOLATION" in res.output_summary:
        st.error(res.output_summary)
    else:
        st.code(res.output_summary, language="text")

st.divider()

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why should an LLM never be given direct read/write credentials to a production database?",
     "Even with strict prompt instructions, models can hallucinate destructive SQL queries (e.g. DROP TABLE or inadvertent UPDATE statements without a WHERE clause) or fall victim to prompt injection attacks. Always enforce security at the database connection layer by using read-only database user credentials and applying static SQL AST parsers."),
    ("How does Tavily differ from traditional Google or Bing search APIs for agentic workflows?",
     "Traditional search APIs return HTML pages, ads, and search engine results pages designed for human browsers. Tavily is purpose-built for LLMs: it scrapes, cleans, extracts raw markdown content, filters out irrelevant advertising and cookie consent dialogs, and returns high-signal factual snippets directly optimized for LLM token ingestion."),
    ("Why are math calculator tools necessary when modern models can generate complex code?",
     "While models understand mathematical formulas conceptually, tokenizers frequently split multi-digit numbers into arbitrary tokens, causing arithmetic hallucinations. A dedicated calculator tool evaluates expressions deterministically with zero token guessing, saving tokens and guaranteeing exact floating-point precision."),
    ("What security measures are required when exposing a Python code execution tool to an agent?",
     "Running arbitrary code generated by an LLM on your host server is dangerous. Code execution tools must run inside ephemeral, isolated sandboxes (e.g. e2b, Modal, or Docker containers with gVisor/Firecracker microVMs) with strict CPU, memory, and timeout limits, read-only file systems, and disabled network access."),
    ("What is the Human-in-the-Loop (HITL) pattern and which tools require it?",
     "HITL pauses agent execution before running destructive or external side-effect operations (e.g. sending emails, deleting records, executing financial transactions) and asks a human operator for explicit approval. Once the human confirms or modifies the action, the agent resumes execution.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Directory Traversal in File Tools. Failing to validate that file paths resolve within a dedicated sandbox root directory allows adversarial inputs to read sensitive files (e.g. ../../etc/shadow or private SSH keys). Always enforce realpath containment checks.")
    st.error("**Trap 2 (Breaking Error)**: Leaking API Keys in Tool Error Messages. When a third-party API tool fails due to an HTTP 401 error, printing raw traceback headers can expose secret API bearer tokens into the conversation history. Always sanitize error logs.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Unbounded Web Scrapes in Search Tools. Allowing tools to return entire un-summarized 50KB HTML pages floods the agent's context window, increasing inference costs and degrading model reasoning. Always truncate and summarize tool outputs.")
    st.warning("**Trap 4 (Latency Warning)**: Long-Running Database Queries. Executing un-indexed analytical SQL queries without a strict statement_timeout blocks the agent worker thread. Always enforce aggressive database query timeouts (e.g. 5–10 seconds).")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Read-Only Database Governance"):
        st.markdown("**Connect with dedicated read-only credentials and enforce static SQL AST validation.**")
    with st.expander("🎴 Flashcard 2: Tavily Search Advantage"):
        st.markdown("**Returns clean markdown snippets optimized for LLM token ingestion, stripping ads and HTML.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Ephemeral Code Sandboxing"):
        st.markdown("**Execute agent-generated Python in isolated microVMs (e2b / Docker) with resource quotas.**")
    with st.expander("🎴 Flashcard 4: Human-in-the-Loop (HITL)"):
        st.markdown("**Mandatory approval gate for operations producing destructive real-world side effects.**")

st.divider()

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I know how to integrate web search tools (Tavily / DuckDuckGo) for live grounding.", value=True)
chk2 = st.checkbox("I understand how to protect database tools using read-only credentials and SQL parsers.", value=True)
chk3 = st.checkbox("I can provide calculator tools for exact deterministic mathematical computations.", value=True)
chk4 = st.checkbox("I understand directory traversal defenses for sandboxed file system tools.", value=True)
chk5 = st.checkbox("I know how to apply Human-in-the-Loop gates to tools with external side effects.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 real-world tools interview questions without reference notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 8.4 Real-World Tools and completed all of Chapter 8: Tools!")
