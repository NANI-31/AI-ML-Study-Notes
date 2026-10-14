import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="8.2 Creating Tools - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("8.2 Creating Tools: The @tool Decorator, Pydantic Schemas, Async & BaseTool")
st.caption("Chapter 8: Tools | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
Building production-grade tools in LangChain ranges from rapid prototyping using the `@tool` decorator 
to engineering mission-critical, asynchronous, stateful tools by subclassing `BaseTool`.

In this lesson, you will master the complete spectrum of tool authoring techniques:
1. **The `@tool` Decorator**: Wrapping standard Python functions with automatic schema generation.
2. **Python Functions as Tools**: Deriving tool names and arguments directly from Python function signatures.
3. **Tool Arguments**: Managing required vs optional arguments and default values.
4. **Tool Descriptions**: Extracting semantic guidance from docstrings or explicit decorator parameters.
5. **Typed Arguments**: Leveraging Python type hints (`int`, `str`, `List[dict]`) for automatic validation.
6. **Pydantic Schemas**: Binding explicit `args_schema` classes for complex validation and boundary rules.
7. **Async Tools**: Building high-throughput coroutine tools (`async def`) for non-blocking I/O operations.
8. **Custom Tools**: Subclassing `BaseTool` to maintain internal state, database connections, and custom lifecycles.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockToolAuthoringResult:
    """Represents the compiled metadata of an authored tool in module scope."""
    def __init__(self, authoring_method: str, tool_name: str, description: str, args_repr: str, is_async: bool):
        self.authoring_method = authoring_method
        self.tool_name = tool_name
        self.description = description
        self.args_repr = args_repr
        self.is_async = is_async

class MockToolAuthoringStudio:
    """Simulates tool authoring compilation, schema extraction, and mock execution in module scope."""
    def __init__(self):
        pass

    def compile_tool(self, method: str, func_name: str, docstring: str, is_async: bool) -> MockToolAuthoringResult:
        name_clean = func_name.strip() if func_name.strip() else "custom_tool"
        desc_clean = docstring.strip() if docstring.strip() else "No description provided."

        if method == "@tool (Basic Decorator)":
            args_repr = "{'query': {'type': 'string', 'description': 'Search query text'}}"
        elif method == "@tool with Pydantic args_schema":
            args_repr = "{\n  'domain': {'type': 'string', 'enum': ['github', 'jira', 'confluence']},\n  'max_results': {'type': 'integer', 'minimum': 1, 'maximum': 50, 'default': 10}\n}"
        else:  # Custom BaseTool Subclass
            args_repr = "{\n  'connection_url': {'type': 'string'},\n  'sql_query': {'type': 'string'},\n  'read_only': {'type': 'boolean', 'default': true}\n}"

        return MockToolAuthoringResult(
            authoring_method=method,
            tool_name=name_clean,
            description=desc_clean,
            args_repr=args_repr,
            is_async=is_async
        )

# --- SECTION 1: ARCHITECTURE BLUEPRINT & TOOL AUTHORING PIPELINE ---
st.header("1. Architecture Blueprint: Tool Authoring Strategies & Decorator Pipeline", divider="rainbow")

st.markdown("""
LangChain offers two primary paradigms for tool authoring: the **Functional `@tool` Decorator** 
(for lightweight, stateless tasks) and the **Object-Oriented `BaseTool` Subclass** (for complex, stateful enterprise services). 
Both compile down to the same underlying `BaseTool` interface.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph G {
        bgcolor="#0B1120";
        rankdir=TB;
        splines=spline;
        node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B", arrowsize=0.8, penwidth=1.2];

        // Level 1: Authoring Approaches
        subgraph cluster_authoring {
            label = "1. AUTHORING PARADIGMS";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            FunctionalApproach [label="Functional Pattern:\\n@tool decorator\\nInspects Python signature &amp; docstrings", color="#10B981"];
            ClassApproach [label="Class-Based Pattern:\\nclass MyTool(BaseTool):\\nExplicit attributes &amp; lifecycle state", color="#38BDF8"];
        }

        // Level 2: Schema Compilation
        subgraph cluster_schema_gen {
            label = "2. SCHEMA INFERENCE &amp; COMPILATION";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#0F172A"];
            SignatureInspector [label="inspect.signature()\\nExtracts argument names, defaults, &amp; type hints"];
            DocstringParser [label="Docstring Parser (Google / Sphinx)\\nExtracts parameter descriptions"];
            PydanticBinder [label="Pydantic args_schema Compiler\\nGenerates OpenAPI parameters schema"];
        }

        // Level 3: Runtime Execution Modes
        subgraph cluster_runtime {
            label = "3. RUNTIME DISPATCH MODES";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            SyncExecution [label="Synchronous Dispatch\\ntool._run(**kwargs)\\nBlocks caller thread"];
            AsyncExecution [label="Asynchronous Dispatch\\ntool._arun(**kwargs)\\nNon-blocking asyncio event loop", color="#10B981"];
        }

        // Level 4: Unified LCEL Component
        subgraph cluster_unified {
            label = "4. UNIFIED LCEL TOOL INSTANCE";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            UnifiedBaseTool [label="Compiled BaseTool Instance\\nname, description, args_schema\\nRunnable Interface Compatible", color="#10B981"];
        }

        // Wiring
        FunctionalApproach -> SignatureInspector;
        FunctionalApproach -> DocstringParser;
        ClassApproach -> PydanticBinder;
        
        SignatureInspector -> UnifiedBaseTool;
        DocstringParser -> UnifiedBaseTool;
        PydanticBinder -> UnifiedBaseTool;

        UnifiedBaseTool -> SyncExecution [label="invoke()"];
        UnifiedBaseTool -> AsyncExecution [label="ainvoke()"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
**Key Authoring Takeaway:**
Whether defined via `@tool` or `BaseTool`, all tools produce a standard `args_schema` Pydantic model and 
implement the LCEL `Runnable` contract, allowing direct piping and asynchronous streaming.
""")

st.divider()

# --- SECTION 2: THE @tool DECORATOR ---
st.header("2. The `@tool` Decorator (Zero-Boilerplate Creation)", divider="rainbow")

st.markdown("""
The `@tool` decorator is the fastest way to turn an ordinary Python function into a LangChain tool. 
It automatically extracts the function name as `name`, the docstring as `description`, 
and the type annotations as `args_schema`.
""")

st.subheader("Basic `@tool` Implementation")
st.code('''from langchain_core.tools import tool

@tool
def calculate_compound_interest(principal: float, annual_rate: float, years: int) -> float:
    """Calculate the future value of an investment using compound interest.
    
    Args:
        principal: Initial investment amount in USD.
        annual_rate: Annual interest rate as a decimal (e.g. 0.05 for 5%).
        years: Number of years the investment compounds.
    """
    return round(principal * ((1 + annual_rate) ** years), 2)

# Inspect the compiled tool attributes:
print(calculate_compound_interest.name)
# 'calculate_compound_interest'

print(calculate_compound_interest.description)
# 'Calculate the future value of an investment using compound interest.'

print(calculate_compound_interest.args)
# {'principal': {'title': 'Principal', 'type': 'number'}, ...}
''', language="python")

st.divider()

# --- SECTION 3: PYTHON FUNCTIONS AS TOOLS ---
st.header("3. Python Functions as Tools (Docstrings & Name Overrides)", divider="rainbow")

st.markdown("""
By default, `@tool` derives the tool name from the Python function's `__name__`. 
However, you can explicitly override the name, description, and return behavior using decorator arguments.
""")

st.code('''from langchain_core.tools import tool

# Customizing tool name and description via decorator arguments
@tool("corporate_directory_search", return_direct=False)
def search_directory(query: str) -> str:
    """Search for corporate employees by name, department, or office location."""
    return f"Found 3 directory matches for query '{query}'."

print(search_directory.name)
# 'corporate_directory_search' (overrode search_directory)
''', language="python")

st.divider()

# --- SECTION 4: TOOL ARGUMENTS ---
st.header("4. Tool Arguments (Required, Optional & Default Values)", divider="rainbow")

st.markdown("""
LangChain tools cleanly distinguish between **strictly required arguments** and **optional parameters with default values**. 
When translated into JSON Schema:
- Arguments without defaults are added to the JSON Schema `required` list.
- Arguments with default values are assigned `default` attributes and omitted from `required`.
""")

st.code('''from langchain_core.tools import tool
from typing import Optional

@tool
def fetch_cloud_logs(
    service_name: str,                      # REQUIRED: No default value
    log_level: str = "ERROR",               # OPTIONAL: Default value 'ERROR'
    limit: int = 50,                        # OPTIONAL: Default value 50
    start_time: Optional[str] = None        # OPTIONAL: Nullable with default None
) -> str:
    """Fetch structured logs from centralized log cluster."""
    return f"Retrieved {limit} '{log_level}' logs for '{service_name}'."

# Schema shows only service_name is required:
print(fetch_cloud_logs.args_schema.schema()["required"])
# ['service_name']
''', language="python")

st.divider()

# --- SECTION 5: TOOL DESCRIPTIONS ---
st.header("5. Tool Descriptions (Docstring Formatting & Semantic Guidance)", divider="rainbow")

st.markdown("""
LangChain parses the docstring of a decorated function to populate parameter descriptions in the JSON Schema. 
It supports standard Google-style, Sphinx, and NumPy docstring conventions.
""")

st.subheader("Comparison of Docstring Formats")
doc_df = pd.DataFrame({
    "Format Style": ["Google Style", "Sphinx Style", "NumPy Style"],
    "Example Parameter Syntax": ["Args:\\n    query (str): The search query", ":param query: The search query", "Parameters\\n----------\\nquery : str\\n    The search query"],
    "LangChain Compatibility": ["Native (Recommended)", "Supported", "Supported"]
})
st.dataframe(doc_df, use_container_width=True)

st.code('''@tool
def trigger_backup(database_name: str, retention_days: int = 7) -> str:
    """Trigger an immediate cold snapshot backup of a relational database.
    
    Args:
        database_name: Canonical identifier of database (e.g. 'users_prod_db').
        retention_days: Number of days before the snapshot is purged. Defaults to 7.
    """
    return f"Snapshot of '{database_name}' initiated with retention {retention_days} days."
''', language="python")

st.divider()

# --- SECTION 6: TYPED ARGUMENTS ---
st.header("6. Typed Arguments (Type Hints & Automatic Coercion)", divider="rainbow")

st.markdown("""
Tools must always specify Python type annotations. Without type annotations, LangChain defaults parameter 
types to open-ended strings (`{"type": "string"}`), preventing the model from emitting integers, booleans, or lists.
""")

st.code('''from langchain_core.tools import tool
from typing import List, Dict

@tool
def aggregate_telemetry_batch(
    node_ids: List[str],             # Maps to JSON Schema type 'array' of 'string'
    metrics: Dict[str, float],       # Maps to JSON Schema type 'object'
    is_live: bool                    # Maps to JSON Schema type 'boolean'
) -> str:
    """Process batched cluster telemetry points."""
    return f"Aggregated {len(node_ids)} nodes with {len(metrics)} metrics. Live={is_live}."
''', language="python")

st.divider()

# --- SECTION 7: PYDANTIC SCHEMAS ---
st.header("7. Pydantic Schemas (`args_schema` Binding)", divider="rainbow")

st.markdown("""
For complex tools requiring multi-field cross-validation, regex validation, or nested parameters, 
relying solely on function arguments is insufficient. 
You can pass a dedicated Pydantic model directly to `@tool(args_schema=...)`.
""")

st.code('''from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool

class DatabaseQueryInput(BaseModel):
    query: str = Field(description="PostgreSQL SELECT query statement")
    read_only: bool = Field(default=True, description="Enforce read-only execution")
    timeout_ms: int = Field(default=5000, ge=100, le=30000, description="Query timeout limit")

    @field_validator("query")
    @classmethod
    def validate_select_only(cls, v: str) -> str:
        clean = v.strip().upper()
        if not clean.startswith("SELECT"):
            raise ValueError("Only read-only SELECT queries are permitted.")
        if any(banned in clean for banned in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]):
            raise ValueError("Mutating operations are strictly forbidden.")
        return v

@tool(args_schema=DatabaseQueryInput)
def execute_safe_sql(query: str, read_only: bool = True, timeout_ms: int = 5000) -> str:
    """Execute a guarded read-only analytical SQL query against the database."""
    return f"Executed query '{query}' in {timeout_ms}ms with read_only={read_only}."
''', language="python")

st.divider()

# --- SECTION 8: ASYNC TOOLS ---
st.header("8. Async Tools (`async def` & Non-Blocking Coroutines)", divider="rainbow")

st.markdown("""
In production web servers (such as FastAPI), AI agents serve multiple concurrent users. 
Synchronous tools block Python's worker thread during database or HTTP calls. 
Defining an async tool allows the agent to await external I/O concurrently without stalling other requests.
""")

st.code('''from langchain_core.tools import tool
import asyncio

# Modern LangChain automatically handles both sync and async via a single async function:
@tool
async def fetch_weather_async(city: str) -> str:
    """Fetch real-time weather asynchronously using non-blocking HTTP requests.
    
    Args:
        city: City name (e.g. 'San Francisco', 'London').
    """
    # Simulate non-blocking network I/O
    await asyncio.sleep(0.5)
    return f"Weather in {city}: 18°C, Partly Cloudy with Light Breeze."

# In an async application:
# observation = await fetch_weather_async.ainvoke({"city": "San Francisco"})
''', language="python")

st.divider()

# --- SECTION 9: CUSTOM TOOLS ---
st.header("9. Custom Tools (Subclassing `BaseTool`)", divider="rainbow")

st.markdown("""
When a tool requires persistent internal state (such as an open database connection pool, Redis cache client, 
or rate-limiting semaphore), subclassing `BaseTool` is the recommended object-oriented pattern.
""")

st.code('''from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Any

class RedisCacheInput(BaseModel):
    key: str = Field(description="Cache key string to lookup")

class RedisCacheReaderTool(BaseTool):
    name: str = "read_redis_cache"
    description: str = "Retrieve cached JSON payloads from Redis by key."
    args_schema: Type[BaseModel] = RedisCacheInput

    # Stateful attributes maintained on the instance:
    cache_ttl_seconds: int = 3600
    host: str = "redis.internal.cluster"

    def _run(self, key: str) -> str:
        """Synchronous cache retrieval."""
        return f"Cache hit for key '{key}' on {self.host}: {{'status': 'cached'}}"

    async def _arun(self, key: str) -> str:
        """Asynchronous cache retrieval."""
        return self._run(key)
''', language="python")

st.divider()

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: Enterprise Production Tool Creation Suite", divider="rainbow")

st.markdown("""
The following complete production pattern demonstrates a comprehensive tool authoring suite 
featuring Pydantic schema validation, async coroutine handlers, input sanitization, and structured telemetry.
""")

st.code('''import asyncio
import logging
from typing import Dict, Any, Type, Optional
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import BaseTool

logger = logging.getLogger("EnterpriseToolSuite")

class GitHubIssueInput(BaseModel):
    repository: str = Field(description="GitHub repository in owner/repo format e.g. 'langchain-ai/langchain'")
    title: str = Field(description="Descriptive issue title", min_length=5, max_length=150)
    body: str = Field(description="Detailed markdown issue body")
    labels: list[str] = Field(default_factory=list, description="Categorization labels")

    @field_validator("repository")
    @classmethod
    def validate_repo_format(cls, v: str) -> str:
        if "/" not in v or len(v.split("/")) != 2:
            raise ValueError("Repository must be formatted as 'owner/repo'.")
        return v.strip().lower()

class GitHubIssueCreatorTool(BaseTool):
    name: str = "create_github_issue"
    description: str = """Create a new issue on GitHub. 
Use when the user explicitly requests filing a bug report or feature proposal.
Requires valid owner/repo format."""
    args_schema: Type[BaseModel] = GitHubIssueInput

    # Stateful API client token
    api_token: Optional[str] = None

    def _run(self, repository: str, title: str, body: str, labels: list[str] = None) -> str:
        labels = labels or []
        logger.info(f"Creating GitHub issue on '{repository}': '{title}'")
        # In real production: invoke GitHub REST API via httpx or PyGithub
        return f"Issue #142 created successfully on {repository}: https://github.com/{repository}/issues/142"

    async def _arun(self, repository: str, title: str, body: str, labels: list[str] = None) -> str:
        logger.info(f"Async dispatch: Creating GitHub issue on '{repository}'")
        await asyncio.sleep(0.1)  # Non-blocking async simulation
        return self._run(repository, title, body, labels)
''', language="python")

st.divider()

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: Tool Authoring & Sandbox Execution Studio", divider="rainbow")

st.markdown("""
Author a tool dynamically below. Choose between `@tool` decoration and `BaseTool` subclassing, 
configure input schemas, and inspect the compiled JSON Schema and execution behavior.
""")

studio = MockToolAuthoringStudio()

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("🛠️ Tool Authoring Configuration")
    chosen_method = st.selectbox(
        "Authoring Method",
        ["@tool (Basic Decorator)", "@tool with Pydantic args_schema", "Custom BaseTool Subclass"]
    )

    func_name_input = st.text_input("Tool Function Name", value="query_knowledge_base")
    docstring_input = st.text_area(
        "Docstring / Tool Description",
        value="Search internal company documentation for engineering runbooks and architectural specs.",
        height=90
    )
    is_async_toggle = st.checkbox("Enable Async Coroutine (_arun / async def)", value=True)

with col_wb2:
    st.subheader("📋 Compiled Tool Metadata")
    compiled = studio.compile_tool(chosen_method, func_name_input, docstring_input, is_async_toggle)
    
    st.info(f"""
    **Tool Name**: `{compiled.tool_name}`  
    **Execution Type**: `{'Asynchronous (Coroutine)' if compiled.is_async else 'Synchronous (Blocking)'}`  
    **Authoring Paradigm**: `{compiled.authoring_method}`
    """)
    st.caption("**Description Registered**:")
    st.write(compiled.description)

st.write("#### 🔍 Generated OpenAPI / JSON Schema Parameters:")
st.code(compiled.args_repr, language="json")

st.divider()

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("When should you subclass BaseTool instead of using the @tool decorator?",
     "Subclass BaseTool when your tool requires: 1) Persistent state (e.g. keeping a database connection pool or API client instance open), 2) Complex custom lifecycle hooks or logging, 3) Distinct implementations for synchronous _run() and asynchronous _arun(), or 4) Dynamic runtime configuration via class initialization arguments."),
    ("How does LangChain populate parameter descriptions in the tool's JSON Schema when using @tool?",
     "LangChain parses the function's docstring using standard docstring parsers (such as Google docstring format). It matches parameter names in the 'Args:' or 'Parameters' section to the function signature and populates the 'description' field of each parameter in the JSON Schema."),
    ("What happens if you define an async tool with async def and invoke it using the synchronous .invoke() method?",
     "When a tool is defined with async def, calling .invoke() will attempt to run the coroutine using an event loop runner (asyncio.run() in a worker thread). However, in nested event loop environments (such as Jupyter notebooks or Tornado/FastAPI), this can raise 'RuntimeError: This event loop is already running'. Always use .ainvoke() for async tools in async applications."),
    ("Why should tool functions always include type annotations on every argument?",
     "Without type annotations, LangChain cannot infer the JSON Schema data type and defaults every parameter to a generic string. This prevents the LLM from passing integers, booleans, or structured arrays, leading to type errors and failed tool executions."),
    ("What is the purpose of passing args_schema to the @tool decorator?",
     "args_schema allows you to bind a custom Pydantic model to a decorated function. This enables advanced Pydantic features such as custom field validators (@field_validator), regex patterns, numerical boundaries (ge, le), and multi-field cross-validation rules before the function executes.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Missing Docstrings on @tool Functions. If a function decorated with @tool lacks a docstring, the tool's description defaults to an empty string. The LLM will have zero semantic information about what the tool does and will rarely or never invoke it.")
    st.error("**Trap 2 (Breaking Error)**: Event Loop Deadlocks in Synchronous Tool Wrappers. Calling asyncio.run() inside a synchronous _run() method while running inside a FastAPI or Tornado async web server causes immediate event loop collision. Always implement native _arun().")
with trap_c2:
    st.warning("**Trap 3 (Validation Warning)**: Pydantic v1 vs v2 Syntax in args_schema. Mixing Pydantic v1 syntax (@validator) inside modern LangChain (which requires Pydantic v2 @field_validator) results in silent validation bypass or runtime deprecation errors.")
    st.warning("**Trap 4 (Quality Warning)**: Overly Complex Argument Signatures. Tools with 8+ arguments confuse models, resulting in high parameter hallucination rates. Keep tool signatures concise, with at most 3 to 4 focused parameters.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: @tool Decorator Function"):
        st.markdown("**Turns a Python function into a BaseTool, extracting schema from annotations and docstrings.**")
    with st.expander("🎴 Flashcard 2: args_schema Parameter"):
        st.markdown("**Binds an explicit Pydantic BaseModel to enforce strict argument validation and bounds.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Async Tool Dispatch"):
        st.markdown("**Implement _arun() or async def and invoke via .ainvoke() for non-blocking I/O.**")
    with st.expander("🎴 Flashcard 4: BaseTool Subclassing"):
        st.markdown("**Used when tools require stateful resources like open DB connections or API tokens.**")

st.divider()

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can create tools using the @tool decorator and Google-style docstrings.", value=True)
chk2 = st.checkbox("I know how to bind custom Pydantic schemas using args_schema.", value=True)
chk3 = st.checkbox("I can implement asynchronous tools using async def and _arun().", value=True)
chk4 = st.checkbox("I know when and how to subclass BaseTool for stateful enterprise operations.", value=True)
chk5 = st.checkbox("I understand how required vs optional parameters are mapped into JSON Schema.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 tool creation interview questions without reference material.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 8.2 Creating Tools. Ready for 8.3 Tool Calling!")
