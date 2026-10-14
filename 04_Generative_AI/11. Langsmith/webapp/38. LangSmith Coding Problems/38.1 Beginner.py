import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="38.1 Beginner Coding Problems - LangSmith Curriculum",
    page_icon="🌱",
    layout="wide"
)

# Custom Theme Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .modern-callout {
        background-color: #0B192C;
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 16px 0;
    }
    .badge-code {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-status {
        background-color: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class LangSmithBeginnerEngine:
    """
    Simulates beginner LangSmith operations:
    Tracing configuration, project isolation, dataset creation,
    example ingestion, and chain execution traces.
    """
    def __init__(self):
        self.projects = {
            "default": {"description": "Default catch-all project", "runs_count": 142, "status": "ACTIVE"},
            "marketing-copywriter-dev": {"description": "Ad copy generation experiments", "runs_count": 48, "status": "ACTIVE"},
            "customer-support-qa": {"description": "Support chat QA testing", "runs_count": 215, "status": "ACTIVE"}
        }
        
        self.datasets = {
            "faq-benchmark-v1": {
                "data_type": "kv",
                "description": "General customer FAQ benchmarks",
                "examples": [
                    {"inputs": {"question": "How do I cancel my trial?"}, "outputs": {"answer": "Go to Account > Billing and click Cancel Trial."}},
                    {"inputs": {"question": "Where is my invoice?"}, "outputs": {"answer": "Invoices are available under Account > Invoices tab."}}
                ]
            }
        }

    def create_project(self, name: str, description: str) -> Dict[str, Any]:
        clean_name = name.strip().lower().replace(" ", "-")
        self.projects[clean_name] = {
            "description": description,
            "runs_count": 0,
            "status": "ACTIVE"
        }
        return {"project_name": clean_name, "status": "CREATED"}

    def add_dataset_example(self, ds_name: str, question: str, answer: str) -> bool:
        if ds_name in self.datasets:
            self.datasets[ds_name]["examples"].append({
                "inputs": {"question": question},
                "outputs": {"answer": answer}
            })
            return True
        return False

    def simulate_traced_chain(self, prompt_text: str, tone: str) -> Dict[str, Any]:
        latency_ms = 420 + len(prompt_text) * 2
        tokens_in = len(prompt_text.split()) + 15
        tokens_out = 35 if tone == "Concise" else 85
        
        output_text = f"[{tone} response] Here is the synthesized output for: {prompt_text[:35]}..."
        return {
            "run_id": f"run-{np.random.randint(1000, 9999)}",
            "prompt_tokens": tokens_in,
            "completion_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out,
            "latency_ms": latency_ms,
            "output": output_text
        }

# Instantiate engine
beginner_engine = LangSmithBeginnerEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Tracing Initialization, Project Workspaces, & Dataset Construction", divider="rainbow")

st.markdown("""
Mastering LangSmith begins with **three foundational pillars**: enabling non-invasive telemetry tracing, 
organizing telemetry runs into segregated project environments, and curating version-controlled benchmark datasets. 
This architectural blueprint illustrates how these components interconnect.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Beginner_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_env {
            label = "Environment Configuration & Telemetry Activation";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";

            env_vars [label="Environment Configuration\\nLANGSMITH_TRACING=true\\nLANGSMITH_API_KEY=lsv2_pt_...\\nLANGSMITH_PROJECT=marketing-dev", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            sdk_init [label="LangSmith Client Initialization\\nfrom langsmith import Client, traceable", fillcolor="#0F172A", fontcolor="#38BDF8"];
        }

        subgraph cluster_projects {
            label = "Project Isolation & Organization";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";

            project_router [label="Project Namespace Router\\n(Directs Traces by Environment)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            proj_dev [label="Project: dev-testing", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            proj_prod [label="Project: production-live", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
        }

        subgraph cluster_datasets {
            label = "Benchmark Dataset Management";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";

            create_ds [label="Dataset Creation\\nclient.create_dataset(name, data_type='kv')", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            add_examples [label="Example Ingestion\\nclient.create_examples(inputs, outputs)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
        }

        subgraph cluster_execution {
            label = "Chain Execution & Traced Spans";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";

            traced_chain [label="@traceable Python Chain / LCEL\\n(Automatic Run Span Creation)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            run_tree [label="Root Run & Child Spans\\n(Inputs, Outputs, Tokens, Latency)", fillcolor="#0F172A", fontcolor="#38BDF8"];
        }

        env_vars -> sdk_init;
        sdk_init -> project_router;
        project_router -> proj_dev;
        project_router -> proj_prod;

        sdk_init -> create_ds;
        create_ds -> add_examples;

        proj_dev -> traced_chain [label="routes runs into"];
        traced_chain -> run_tree [label="emits telemetry"];
        add_examples -> traced_chain [label="benchmark inputs"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Best Practice Notice:</strong> Always set <code>LANGSMITH_PROJECT</code> explicitly in your environment or code. 
    If omitted, all traces fall back to the <code>default</code> project, mixing local developer debugging with production monitoring.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: ENABLE TRACING FOR AN LLM APPLICATION
# ==============================================================================
st.header("2. Enable Tracing for an LLM Application: Environment Variables and SDK Configuration", divider="rainbow")

st.markdown("""
LangSmith is designed for zero-code-change telemetry when using standard LangChain or LangGraph components. 
Setting environment variables activates the background telemetry client automatically.
""")

st.markdown("""
### The Three Required Environment Variables
To enable LangSmith tracing across any Python process:
""")

st.code('''# Set in .env, shell profile, or container configuration:
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="lsv2_pt_your_unique_api_key_here"
export LANGSMITH_PROJECT="my-first-langsmith-project"

# Optional: Custom endpoint for self-hosted / enterprise instances
# export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
''', language="bash")

st.markdown("""
### Programmatic Verification in Python
You can also inspect or set these variables programmatically before instantiating LLM clients:
""")

st.code('''# Programmatic check without crashing
import os

def verify_langsmith_config():
    is_tracing = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"
    api_key_set = bool(os.environ.get("LANGSMITH_API_KEY"))
    project_name = os.environ.get("LANGSMITH_PROJECT", "default")
    
    print(f"Tracing Enabled: {is_tracing}")
    print(f"API Key Present: {api_key_set}")
    print(f"Target Project:  {project_name}")

verify_langsmith_config()
''', language="python")

# ==============================================================================
# SECTION 3: CREATE A LANGSMITH PROJECT
# ==============================================================================
st.header("3. Create a LangSmith Project: Scoping, Metadata Tags, and Retention", divider="rainbow")

st.markdown("""
A **Project** in LangSmith is a logical namespace that groups related traces. 
Projects allow engineering teams to isolate environments (e.g., `dev`, `staging`, `prod`), partition 
independent features, and set distinct trace retention and access policies.
""")

st.markdown("""
### Programmatic Project Creation with `langsmith.Client`
""")

st.code('''from langsmith import Client

client = Client()

PROJECT_NAME = "ecommerce-recommendation-prod"

# Check if project exists, or create with metadata description
if not client.has_project(project_name=PROJECT_NAME):
    new_project = client.create_project(
        project_name=PROJECT_NAME,
        description="Production traces for user recommendation agent",
        metadata={
            "environment": "production",
            "team": "search-and-discovery",
            "tier": "tier-1-critical"
        }
    )
    print(f"Project created with ID: {new_project.id}")
else:
    print(f"Project '{PROJECT_NAME}' already exists.")
''', language="python")

# ==============================================================================
# SECTION 4: CREATE A DATASET
# ==============================================================================
st.header("4. Create a Dataset: Dataset Types, Schemas, and Versioning", divider="rainbow")

st.markdown("""
A **Dataset** is a collection of benchmark examples used for offline evaluation, prompt benchmarking, 
and CI/CD regression testing. LangSmith supports three primary dataset formats:
""")

ds_col1, ds_col2, ds_col3 = st.columns(3)
with ds_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>1. KV (Key-Value) Dataset</h4>
        <span class="badge-code">data_type="kv"</span>
        <p>Flexible dictionary inputs and outputs. Standard for RAG pipelines, extraction chains, and general chains.</p>
        <p style="color: #94A3B8; font-size: 12px;">Example: <code>{"inputs": {"q": "..."}}</code></p>
    </div>
    """, unsafe_allow_html=True)

with ds_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>2. Chat Dataset</h4>
        <span class="badge-code">data_type="chat"</span>
        <p>Contains message arrays representing multi-turn conversational exchanges (System, Human, AI).</p>
        <p style="color: #94A3B8; font-size: 12px;">Standard for chatbots and roleplaying.</p>
    </div>
    """, unsafe_allow_html=True)

with ds_col3:
    st.markdown("""
    <div class="metric-card">
        <h4>3. LLM (Legacy Text) Dataset</h4>
        <span class="badge-code">data_type="llm"</span>
        <p>Raw string prompt inputs mapped to string completion outputs.</p>
        <p style="color: #94A3B8; font-size: 12px;">Used for raw completion models.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### Creating a KV Dataset via Python SDK:")

st.code('''from langsmith import Client

client = Client()

DATASET_NAME = "customer-support-golden-v1"

if not client.has_dataset(dataset_name=DATASET_NAME):
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Golden Q&A benchmark for customer support tier-1 issues.",
        data_type="kv"
    )
    print(f"Dataset created: {dataset.name} (ID: {dataset.id})")
''', language="python")

# ==============================================================================
# SECTION 5: ADD EXAMPLES TO A DATASET
# ==============================================================================
st.header("5. Add Examples to a Dataset: Single Examples, Batch Ingestion, and Splits", divider="rainbow")

st.markdown("""
Once a dataset container is created, you populate it with **Examples**. 
Each example consists of:
- `inputs`: The dictionary of parameters passed into your application.
- `outputs`: The expected ground truth dictionary used by evaluators.
- `metadata` (optional): Extra tags like test splits (`train`, `test`, `edge-cases`).
""")

st.markdown("### Method 1: Single Example Ingestion")

st.code('''client.create_example(
    inputs={"question": "What is your refund policy for hardware?"},
    outputs={"expected_answer": "Hardware refunds are accepted within 30 days in original packaging."},
    dataset_name=DATASET_NAME,
    metadata={"category": "billing", "difficulty": "easy"}
)
''', language="python")

st.markdown("### Method 2: Batch Ingestion via `create_examples`")

st.code('''questions = [
    "How do I reset my password?",
    "Where do I find my API key?",
    "Can I invite team members to my organization?"
]

expected_answers = [
    "Click 'Forgot Password' on the login screen to receive a reset link.",
    "Go to Settings > Developer Settings > API Keys to copy your token.",
    "Navigate to Organization > Members and click 'Invite Member'."
]

# Batch upload to LangSmith
client.create_examples(
    inputs=[{"question": q} for q in questions],
    outputs=[{"expected_answer": a} for a in expected_answers],
    dataset_name=DATASET_NAME
)
print(f"Batch added {len(questions)} examples successfully!")
''', language="python")

# ==============================================================================
# SECTION 6: TRACE A SIMPLE CHAIN
# ==============================================================================
st.header("6. Trace a Simple Chain: The `@traceable` Decorator and Nested Spans", divider="rainbow")

st.markdown("""
While LangChain components trace automatically, production architectures often use standard 
Python code, custom FastAPI endpoints, or proprietary vector search libraries. 
The modern `@traceable` decorator instruments **any Python function** with automatic run creation, 
timing, and exception tracking.
""")

st.code('''from langsmith import traceable

@traceable(run_type="parser", name="normalize_user_query")
def preprocess_text(text: str) -> str:
    """Sub-span: normalizes and cleans input."""
    return text.strip().lower()

@traceable(run_type="llm", name="mock_llm_synthesizer")
def call_llm(cleaned_query: str) -> str:
    """Sub-span: simulates language model generation."""
    return f"Response to: {cleaned_query}"

@traceable(run_type="chain", name="end_to_end_support_chain")
def run_support_chain(raw_query: str) -> dict:
    """Root span: orchestrates entire execution workflow."""
    clean_q = preprocess_text(raw_query)
    response = call_llm(clean_q)
    return {"query": raw_query, "answer": response}

# Invoking the function automatically pushes a hierarchical trace to LangSmith!
result = run_support_chain("   HOW DO I INVITE A TEAM MEMBER?   ")
print(result)
''', language="python")

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: End-to-End Traced Chain and Dataset Pipeline with LangSmith Client", divider="rainbow")

st.markdown("""
Below is the complete, self-contained Python script integrating project initialization, 
dataset creation, example ingestion, and execution of a traced chain.
""")

st.code('''import os
from langsmith import Client, traceable

# 1. Initialize LangSmith Client
client = Client()

PROJECT_NAME = "beginner-onboarding-demo"
DATASET_NAME = "onboarding-qa-dataset"

# 2. Ensure Project Exists
if not client.has_project(project_name=PROJECT_NAME):
    client.create_project(
        project_name=PROJECT_NAME,
        description="Demo project for LangSmith beginner coding tutorial."
    )
    print(f"Created project: {PROJECT_NAME}")

# 3. Ensure Dataset Exists & Add Examples
if not client.has_dataset(dataset_name=DATASET_NAME):
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Starter benchmark questions."
    )
    client.create_examples(
        inputs=[
            {"question": "What is LangSmith?"},
            {"question": "How do I create a run?"}
        ],
        outputs=[
            {"answer": "LangSmith is a platform for building, testing, and monitoring LLM applications."},
            {"answer": "Runs are created automatically by invoking traced chains or using @traceable."}
        ],
        dataset_id=dataset.id
    )
    print(f"Created dataset and uploaded examples: {DATASET_NAME}")

# 4. Define and Execute Traced Chain
@traceable(
    run_type="chain",
    name="simple_faq_responder",
    project_name=PROJECT_NAME
)
def faq_responder(question: str) -> dict:
    """Processes question and returns simulated response."""
    return {
        "question": question,
        "answer": f"Processed answer for query: {question}"
    }

# Execute query under active project
res = faq_responder("What is LangSmith?")
print("Execution complete:", res)
''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Beginner Tracing, Project Setup & Dataset Manager Simulator", divider="rainbow")

st.markdown("""
Use the interactive console below to test creating LangSmith projects, adding dataset examples, 
and simulating traced chain executions in a sandbox environment.
""")

wb_action = st.selectbox(
    "Choose Workbench Action:",
    ["1. Project Namespace Manager", "2. Dataset & Example Ingestion Console", "3. Traced Chain Execution Sandbox"]
)

if wb_action == "1. Project Namespace Manager":
    st.subheader("Project Inventory & Creation")
    p_df = pd.DataFrame([
        {"Project Name": k, "Description": v["description"], "Runs Count": v["runs_count"], "Status": v["status"]}
        for k, v in beginner_engine.projects.items()
    ])
    st.dataframe(p_df, use_container_width=True)

    st.markdown("#### Create New Project:")
    c1, c2 = st.columns(2)
    with c1:
        new_p_name = st.text_input("Project Name:", "recommendation-dev")
    with c2:
        new_p_desc = st.text_input("Description:", "Dev testing for recommendation model")

    if st.button("Create Project"):
        res = beginner_engine.create_project(new_p_name, new_p_desc)
        st.success(f"Project '{res['project_name']}' created successfully!")
        st.rerun()

elif wb_action == "2. Dataset & Example Ingestion Console":
    st.subheader("Manage Benchmark Datasets")
    d_names = list(beginner_engine.datasets.keys())
    active_ds = st.selectbox("Select Target Dataset:", d_names)

    cur_ds = beginner_engine.datasets[active_ds]
    st.info(f"**Description:** {cur_ds['description']} | **Data Type:** `{cur_ds['data_type']}`")

    examples_df = pd.DataFrame([
        {"Question (Input)": ex["inputs"]["question"], "Answer (Output Ground Truth)": ex["outputs"]["answer"]}
        for ex in cur_ds["examples"]
    ])
    st.dataframe(examples_df, use_container_width=True)

    st.markdown("#### Add New Example:")
    q_in = st.text_input("Input Question:", "How do I upgrade my subscription plan?")
    a_in = st.text_input("Expected Ground Truth Answer:", "Navigate to Account > Billing > Plans and select Upgrade.")

    if st.button("Add Example"):
        if beginner_engine.add_dataset_example(active_ds, q_in, a_in):
            st.success("Example added to dataset successfully!")
            st.rerun()

else:
    st.subheader("Execute and Trace Simple Chain")
    test_prompt = st.text_area("Input Prompt:", "Summarize the key advantages of using LangSmith for LLM tracing.")
    test_tone = st.selectbox("Desired Output Tone:", ["Concise", "Detailed", "Technical"])

    if st.button("Run Traced Chain"):
        sim_res = beginner_engine.simulate_traced_chain(test_prompt, test_tone)
        st.markdown(f"**Run ID:** `{sim_res['run_id']}`")
        st.markdown(f"**Latency:** `{sim_res['latency_ms']}ms` | **Tokens:** `{sim_res['total_tokens']}` ({sim_res['prompt_tokens']} prompt + {sim_res['completion_tokens']} completion)")
        st.success(f"**Generated Output:** {sim_res['output']}")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: What happens if an application invokes an LLM while LANGSMITH_TRACING='true' but no API key is set?"):
    st.markdown("""
    **Answer:**  
    The LangSmith SDK catches authentication failures silently in the background batch thread without crashing 
    the host application. A warning is logged to standard error (`stderr`), but user requests continue to execute 
    unhindered to guarantee application resilience.
    """)

with st.expander("Q2: Why should teams avoid sending all traces to the 'default' LangSmith project?"):
    st.markdown("""
    **Answer:**  
    The `default` project becomes a catch-all dumping ground. Mixing local developer ad-hoc experiments with production 
    traffic pollutes aggregate metrics (latency, token cost, error rates) and makes it impossible to establish 
    trustworthy production SLIs or isolate bugs by environment.
    """)

with st.expander("Q3: What is the fundamental difference between a 'kv' dataset and a 'chat' dataset?"):
    st.markdown("""
    **Answer:**  
    - A **kv dataset** stores generic JSON objects with arbitrary keys in `inputs` and `outputs` (ideal for RAG, extraction, and chains).
    - A **chat dataset** strictly formats `inputs` and `outputs` as standard message arrays containing role specifications (`system`, `human`, `ai`, `tool`).
    """)

with st.expander("Q4: How does the @traceable decorator handle functions that throw unhandled exceptions?"):
    st.markdown("""
    **Answer:**  
    The `@traceable` decorator wraps the target function in an execution context. When an exception is raised, 
    it records the exception type, error message, and full stack trace into the run span's `error` field, marks the 
    run status as `FAILED` / `ERROR`, and then re-raises the exception so application error handling remains intact.
    """)

with st.expander("Q5: Can you trace non-LangChain code (e.g., standard OpenAI SDK calls or custom Python functions)?"):
    st.markdown("""
    **Answer:**  
    Yes. LangSmith is framework-agnostic. By wrapping standard Python functions in `@traceable` or using the 
    LangSmith `RunTree` API, any code (OpenAI SDK, Anthropic SDK, PyTorch models, custom database queries) can be traced.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Hardcoding LANGSMITH_API_KEY into Source Code**  
Committing API keys to Git repositories creates severe security vulnerabilities. Always load API keys 
from secure environment secret managers or `.env` files.
""")

st.error("""
**Trap 2: Creating Duplicate Datasets on Every Pipeline Run**  
Calling `client.create_dataset()` without checking `client.has_dataset()` first causes name collisions 
or creates hundreds of redundant dataset instances.
""")

st.warning("""
**Trap 3: Forgetting to Specify run_type in @traceable**  
Decorating functions with `@traceable` without setting `run_type="retriever"`, `run_type="llm"`, or `run_type="parser"` 
causes LangSmith to classify all spans as generic chains, degrading visualization in the UI.
""")

st.warning("""
**Trap 4: Modifying Golden Dataset Examples Without Version Tracking**  
Overwriting benchmark examples without tracking version tags prevents historical regression comparisons.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>LANGSMITH_TRACING</h4>
        <p>The global boolean environment variable that turns on non-blocking background telemetry across Python processes.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Project Namespace</h4>
        <p>A designated logical container in LangSmith that isolates traces by environment, feature, or engineering team.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>KV Dataset</h4>
        <p>A benchmark container storing dictionary inputs and outputs for automated evaluation and regression testing.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>@traceable</h4>
        <p>The LangSmith Python decorator that instruments plain Python functions with hierarchical run telemetry.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I know the three environment variables required to activate LangSmith tracing.", value=True)
chk_2 = st.checkbox("I can create and configure isolated LangSmith projects using client.create_project().", value=True)
chk_3 = st.checkbox("I understand the distinction between KV and Chat dataset types in LangSmith.", value=True)
chk_4 = st.checkbox("I can populate datasets with single and batch examples using create_examples().", value=True)
chk_5 = st.checkbox("I know how to instrument custom Python functions with the @traceable decorator and specify run_type.", value=True)
chk_6 = st.checkbox("I understand how LangSmith captures exceptions and stack traces in failed runs.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Beginner LangSmith coding competencies verified! Ready for Intermediate Coding Problems.")
