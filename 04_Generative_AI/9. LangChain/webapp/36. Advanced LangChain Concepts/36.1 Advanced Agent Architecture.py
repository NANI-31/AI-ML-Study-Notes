import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="36.1 Advanced Agent Architecture - LangChain Curriculum",
    page_icon="🧠",
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
    .arch-badge {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-route {
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
class AdvancedAgentArchitectureEngine:
    """
    Simulates advanced multi-agent routing, explicit handoffs,
    dynamic tool permission filtering, and middleware interceptors.
    """
    def __init__(self):
        self.agent_directory = {
            "Triage_Supervisor": {"role": "Root Router", "tools": ["transfer_to_billing", "transfer_to_tech"]},
            "Billing_Specialist": {"role": "Financials", "tools": ["lookup_invoice", "issue_refund"]},
            "Tech_Specialist": {"role": "Infrastructure", "tools": ["check_cluster_health", "restart_pod"]}
        }

    def simulate_handoff(self, user_intent: str) -> Dict[str, Any]:
        intent_lower = user_intent.lower()
        if any(w in intent_lower for w in ["refund", "billing", "invoice", "charge", "card"]):
            target_agent = "Billing_Specialist"
            action = "transfer_to_billing(reason='User inquiry regarding financials')"
        elif any(w in intent_lower for w in ["error", "pod", "cluster", "crash", "bug", "server"]):
            target_agent = "Tech_Specialist"
            action = "transfer_to_tech(reason='User inquiry regarding infrastructure fault')"
        else:
            target_agent = "Triage_Supervisor"
            action = "clarify_intent()"

        return {
            "initial_agent": "Triage_Supervisor",
            "detected_intent": user_intent,
            "handoff_tool_invoked": action,
            "active_agent_post_handoff": target_agent,
            "scoped_tools": self.agent_directory[target_agent]["tools"]
        }

    def filter_tools_by_role(self, user_role: str) -> List[str]:
        all_tools = {
            "read_public_docs": "anonymous",
            "submit_ticket": "user",
            "view_user_invoices": "user",
            "issue_refund": "billing_admin",
            "restart_production_cluster": "super_admin"
        }
        allowed = []
        role_hierarchy = {
            "anonymous": 1,
            "user": 2,
            "billing_admin": 3,
            "super_admin": 4
        }
        user_level = role_hierarchy.get(user_role, 1)

        for tool_name, min_role in all_tools.items():
            if user_level >= role_hierarchy[min_role]:
                allowed.append(tool_name)
        return allowed

# Instantiate engine
arch_engine = AdvancedAgentArchitectureEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Advanced Multi-Agent Topology, Hierarchical Routing, & Dynamic Dispatch", divider="rainbow")

st.markdown("""
As AI applications expand in scope, monolithic single-agent systems become brittle, slow, and unreliable. 
**Advanced Agent Architecture** structures applications into **modular multi-agent networks**: 
supervisors route traffic, specialized subagents communicate via explicit **handoffs**, and execution boundaries 
dynamically adjust tools, models, and system prompts based on user permissions and task complexity.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Advanced_Agent_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_ingress {
            label = "Agent Middleware & Ingress Interception Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            in_query [label="Incoming User Query\\n(User token + prompt payload)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            middleware_guard [label="Agent Middleware\\n(PII sanitization, role extraction, token budget checks)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
        }

        subgraph cluster_supervisor {
            label = "Supervisor & Intent Routing Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            triage_agent [label="Triage Supervisor Agent\\n(Lightweight intent classification, zero heavy tools)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            handoff_decision [label="Explicit Agent Handoff\\n(Emits transfer_to_billing or transfer_to_tech)", fillcolor="#064E3B", fontcolor="#34D399", color="#10B981"];
        }

        subgraph cluster_subagents {
            label = "Specialized Domain Subagents (Dynamic Scoping)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            billing_subagent [label="Billing Subagent (GPT-4o-mini)\\n(Dynamic Tools: lookup_invoice, refund)\\n(Dynamic Prompt: Financial policy directives)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            tech_subagent [label="Tech Subagent (Claude 3.5 Sonnet)\\n(Dynamic Tools: check_cluster, restart_pod)\\n(Dynamic Model: Escalates to frontier for code diagnostics)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
        }

        subgraph cluster_response {
            label = "Response Assembly & State Handoff Return";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            final_reply [label="Synthesized Response to User\\n(Thread state checkpointed)", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
        }

        in_query -> middleware_guard;
        middleware_guard -> triage_agent;
        triage_agent -> handoff_decision;

        handoff_decision -> billing_subagent [label="financial intent"];
        handoff_decision -> tech_subagent [label="infrastructure intent"];

        billing_subagent -> final_reply;
        tech_subagent -> final_reply;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>The Handoff Pattern:</strong> Instead of passing entire conversation histories back and forth through a central bottleneck, 
    subagents take ownership of the thread. A handoff function (e.g., <code>transfer_to_billing</code>) transfers control 
    directly to the specialist subagent with scoped tools, dramatically improving task completion accuracy.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: AGENT ROUTING
# ==============================================================================
st.header("2. Agent Routing: Intent Classification and Fallback Routes", divider="rainbow")

st.markdown("""
**Agent Routing** directs user requests to the appropriate domain handler. 
Routing can be implemented via deterministic rule engines or LLM classification.
""")

st.markdown("""
### Routing Architecture Spectrum
- **Deterministic Regex/Keyword Routing**: For high-confidence commands (e.g., `/reset`, `status`), route immediately without an LLM reasoning pass, saving 100% of latency and cost.
- **LLM Intent Classifier**: A lightweight model (GPT-4o-mini or Haiku) parses ambiguous user phrasing and emits a structured classification enum.
- **Fallback Safe Route**: If user intent remains ambiguous after classification, route to a General Triage Assistant with conversational clarification capabilities.
""")

# ==============================================================================
# SECTION 3: AGENT HANDOFFS
# ==============================================================================
st.header("3. Agent Handoffs: Explicit Transfer Functions and Context Bridging", divider="rainbow")

st.markdown("""
In Swarm-style and LangGraph multi-agent architectures, agents switch execution context 
by invoking **explicit handoff tools**.
""")

st.code('''from langchain_core.tools import tool

@tool
def transfer_to_billing(reason: str) -> str:
    \"\"\"Transfers active conversation control to the specialized Billing Agent.\"\"\"
    return f"Handoff executed: {reason}"

@tool
def transfer_to_tech_support(reason: str) -> str:
    \"\"\"Transfers active conversation control to the Infrastructure Tech Agent.\"\"\"
    return f"Handoff executed: {reason}"
''', language="python")

st.markdown("""
When the supervisor model emits a `tool_call` for `transfer_to_billing`, the runtime router 
transitions the graph execution pointer to the `billing_subagent` node, passing along 
the context bridge reason.
""")

# ==============================================================================
# SECTION 4: SUBAGENTS
# ==============================================================================
st.header("4. Subagents: Hierarchical Worker Isolation and Encapsulated Scopes", divider="rainbow")

st.markdown("""
Subagents operate as self-contained micro-agents with **encapsulated state scopes**:
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Why Subagents Outperform Monoliths
    - **Focused Tool Catalog**: Each subagent has 2 to 4 tools instead of 30, eliminating tool selection confusion.
    - **Concise System Prompts**: Prompts are domain-specific (e.g., 200 tokens) rather than omnibus prompts (2,000 tokens).
    - **Isolated Context**: Subagent scratchpad notes do not pollute the parent supervisor's conversation thread.
    """)

with col2:
    st.markdown("""
    ### State Isolation Patterns
    - **Private Subagent State**: Intermediate calculations and raw API dumps remain local to the subagent.
    - **Public Return Value**: Only the final synthesis or handoff output is merged back into the parent thread messages.
    """)

# ==============================================================================
# SECTION 5: DYNAMIC TOOLS
# ==============================================================================
st.header("5. Dynamic Tools: Runtime Tool Binding and Role-Based Permissions", divider="rainbow")

st.markdown("""
Never bind every possible tool to every user. **Dynamic Tool Binding** configures the model's 
available tool catalog at runtime based on the authenticated user's Role-Based Access Control (RBAC) tier.
""")

st.code('''def get_tools_for_request(user_role: str) -> list:
    """Dynamically binds tools based on caller RBAC permissions."""
    base_tools = [search_knowledge_base, check_faq]
    
    if user_role in ["support_agent", "admin"]:
        base_tools.extend([lookup_account, create_ticket])
        
    if user_role == "admin":
        base_tools.append(issue_immediate_refund)
        
    return base_tools

# At runtime:
# user_tools = get_tools_for_request(current_user.role)
# model_with_tools = chat_model.bind_tools(user_tools)
''', language="python")

# ==============================================================================
# SECTION 6: DYNAMIC MODELS
# ==============================================================================
st.header("6. Dynamic Models: Complexity-Based Escalation and Model Fallbacks", divider="rainbow")

st.markdown("""
Different nodes in an agent graph have vastly different reasoning requirements. 
Advanced architectures bind different LLM checkpoints dynamically to different graph nodes:
""")

model_dyn_df = pd.DataFrame([
    {"Graph Node": "triage_supervisor", "Dynamic Model": "GPT-4o-mini (Temp=0.0)", "Reasoning": "Fast, deterministic intent routing and function calling."},
    {"Graph Node": "sql_generation_node", "Dynamic Model": "Claude 3.5 Sonnet (Temp=0.0)", "Reasoning": "High-precision code and complex syntax generation."},
    {"Graph Node": "creative_summary_node", "Dynamic Model": "GPT-4o (Temp=0.7)", "Reasoning": "Rich narrative prose and empathetic customer tone."}
])
st.dataframe(model_dyn_df, use_container_width=True)

# ==============================================================================
# SECTION 7: DYNAMIC PROMPTS
# ==============================================================================
st.header("7. Dynamic Prompts: Few-Shot Selector Injection and Directive Swapping", divider="rainbow")

st.markdown("""
Static prompts cannot adapt to evolving context. **Dynamic Prompting** swaps system directives, 
injects relevant few-shot examples using vector similarity, and binds tenant-specific branding guidelines on the fly.
""")

st.code('''from langchain_core.prompts import ChatPromptTemplate

def build_dynamic_prompt(tenant_tier: str) -> ChatPromptTemplate:
    base_directive = "You are an enterprise support specialist."
    
    if tenant_tier == "VIP_ENTERPRISE":
        directive = f"{base_directive} Priority SLA active. Provide deep technical root-cause analyses."
    else:
        directive = f"{base_directive} Standard SLA. Provide concise self-service resolution links."
        
    return ChatPromptTemplate.from_messages([
        ("system", directive),
        ("placeholder", "{messages}")
    ])
''', language="python")

# ==============================================================================
# SECTION 8: AGENT MIDDLEWARE
# ==============================================================================
st.header("8. Agent Middleware: Intercepting Inputs, Guardrails, and Token Budgets", divider="rainbow")

st.markdown("""
**Agent Middleware** sits between user ingress and model invocation, executing cross-cutting operational concerns:
""")

st.markdown("""
1. **PII Sanitization**: Redacts credit card numbers, Social Security numbers, and email addresses before text hits LLM providers.
2. **Prompt Injection Guardrails**: Evaluates incoming text against safety classifiers to block jailbreak attempts.
3. **Token & Concurrency Throttling**: Verifies the tenant has not breached their monthly token allocation.
4. **Structured Output Enforcement**: Automatically repairs slightly malformed JSON emitted by model tools.
""")

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Full Multi-Agent Handoff and Dynamic Tool Architecture", divider="rainbow")

st.markdown("""
Below is the self-contained production Python script implementing a complete 
**multi-agent handoff supervisor with dynamic tool permission filtering**.
""")

st.code('''from typing import Dict, Any, List

class ProductionMultiAgentOrchestrator:
    def __init__(self):
        self.active_agents = {
            "triage": {"name": "Triage Supervisor", "model": "fast-slm"},
            "billing": {"name": "Billing Specialist", "model": "fast-slm"},
            "tech": {"name": "Infrastructure Engineer", "model": "frontier-llm"}
        }

    def evaluate_middleware(self, prompt: str, user_role: str) -> bool:
        """Middleware security and budget pre-check."""
        # Sanity check: reject empty queries or forbidden words
        if not prompt.strip():
            return False
        return True

    def get_scoped_tools(self, agent_key: str, user_role: str) -> List[str]:
        """Dynamic tool binding based on agent specialization and RBAC."""
        if agent_key == "billing":
            tools = ["lookup_balance"]
            if user_role in ["billing_admin", "super_admin"]:
                tools.append("issue_refund")
            return tools
        elif agent_key == "tech":
            return ["tail_logs", "check_status"]
        return ["route_to_billing", "route_to_tech"]

    def execute_request(self, user_prompt: str, user_role: str = "user") -> Dict[str, Any]:
        # Step 1: Middleware Interception
        if not self.evaluate_middleware(user_prompt, user_role):
            return {"error": "Middleware blocked request."}

        # Step 2: Triage Classification
        lower_p = user_prompt.lower()
        if "refund" in lower_p or "invoice" in lower_p:
            active_agent = "billing"
        elif "server" in lower_p or "cluster" in lower_p:
            active_agent = "tech"
        else:
            active_agent = "triage"

        # Step 3: Dynamic Tool Scoping
        bound_tools = self.get_scoped_tools(active_agent, user_role)

        # Step 4: Agent Execution
        return {
            "prompt": user_prompt,
            "routed_agent": self.active_agents[active_agent]["name"],
            "model_tier": self.active_agents[active_agent]["model"],
            "dynamically_bound_tools": bound_tools,
            "status": "HANDOFF_RESOLVED"
        }

orchestrator = ProductionMultiAgentOrchestrator()
# Test Standard User
r1 = orchestrator.execute_request("I need an invoice refund for last month.", user_role="user")
print("Standard User Run:", r1)

# Test Admin User
r2 = orchestrator.execute_request("I need an invoice refund for last month.", user_role="billing_admin")
print("Billing Admin Run:", r2)
''', language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Agent Handoff Simulator & Dynamic Tool Permission Tester", divider="rainbow")

st.markdown("""
Test how user intent dictates agent handoffs and observe how 
Role-Based Access Control dynamically filters available tool definitions.
""")

wb_view = st.selectbox(
    "Select Architecture Tool:",
    ["1. Agent Intent Handoff Simulator", "2. Dynamic Tool Permission (RBAC) Filter", "3. Dynamic Model Allocation Inspector"]
)

if wb_view == "1. Agent Intent Handoff Simulator":
    st.subheader("Autonomous Handoff Simulation")
    test_user_prompt = st.text_input("Enter User Inquiry:", "My Kubernetes pod is crashing with OutOfMemory errors.")

    if st.button("Dispatch to Triage Supervisor"):
        handoff_data = arch_engine.simulate_handoff(test_user_prompt)

        st.markdown(f"**Originating Agent:** `{handoff_data['initial_agent']}`")
        st.markdown(f"**Action Executed:** `{handoff_data['handoff_tool_invoked']}`")
        st.success(f"**Handoff Complete:** Transferred to `{handoff_data['active_agent_post_handoff']}`")
        st.markdown(f"**Active Tools Available to Specialist:** `{handoff_data['scoped_tools']}`")

elif wb_view == "2. Dynamic Tool Permission (RBAC) Filter":
    st.subheader("Role-Based Dynamic Tool Filtering")
    role_choice = st.selectbox("Select Authenticated User Role:", ["anonymous", "user", "billing_admin", "super_admin"])

    allowed_tools = arch_engine.filter_tools_by_role(role_choice)
    st.markdown(f"**Permitted Tools for `{role_choice}` ({len(allowed_tools)} tools bound):**")

    for t in allowed_tools:
        st.markdown(f"- `tool` **{t}**")

    if "issue_refund" in allowed_tools:
        st.success("Financial refund capabilities unlocked.")
    if "restart_production_cluster" in allowed_tools:
        st.error("DANGEROUS: Infrastructure cluster restart capabilities unlocked.")

else:
    st.subheader("Dynamic Model Allocation Matrix")
    st.dataframe(model_dyn_df, use_container_width=True)
    st.caption("Binding distinct models to specialized graph nodes maximizes reasoning efficiency while cutting costs.")

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: What is the fundamental difference between an Agent Supervisor and explicit Agent Handoffs?"):
    st.markdown("""
    **Answer:**  
    - In an **Agent Supervisor** pattern, all communication passes through a central coordinator. Workers execute tasks and return results back to the supervisor, creating a potential communication bottleneck.
    - In **Agent Handoffs**, a specialist agent transfers ownership of the thread directly to another subagent via tool invocation (e.g., `transfer_to_billing`). The receiving agent directly communicates with the user and owns subsequent turns.
    """)

with st.expander("Q2: Why is dynamic tool binding critical for enterprise multi-tenant security?"):
    st.markdown("""
    **Answer:**  
    If sensitive administrative tools (such as `issue_refund` or `delete_database`) are bound statically to the model, 
    an unprivileged user can trigger those tools through prompt injection or jailbreaking. Dynamic tool binding 
    filters the model's tool schema at the application layer based on validated JWT/RBAC claims, making unauthorized tool 
    invocations physically impossible for the model to execute.
    """)

with st.expander("Q3: How does agent middleware protect production LLM services from malformed outputs?"):
    st.markdown("""
    **Answer:**  
    Agent middleware intercepts responses between the LLM and tool invocation nodes. If a model emits JSON with minor 
    syntax errors (trailing commas, missing quotes), the middleware attempts deterministic regex or Pydantic repair 
    before raising an exception, avoiding unnecessary failed turns.
    """)

with st.expander("Q4: What is the advantage of using dynamic models across different StateGraph nodes?"):
    st.markdown("""
    **Answer:**  
    Different tasks have radically different compute requirements. Intent classification and simple routing are handled 
    optimally by fast, inexpensive SLMs (GPT-4o-mini). Complex coding or policy analysis is routed to frontier models 
    (Claude 3.5 Sonnet). Dynamically assigning models per node ensures optimal quality-to-cost allocation across the pipeline.
    """)

with st.expander("Q5: How do subagents prevent context window pollution in multi-turn conversations?"):
    st.markdown("""
    **Answer:**  
    Subagents operate in isolated execution scopes. An intermediate multi-step search (generating dozens of raw tool 
    messages) occurs strictly inside the subagent's local state. When completed, the subagent returns only a consolidated 
    summary to the parent conversation, preserving the main thread's token budget.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Binding Privileged Administrative Tools to Public Endpoints**  
Exposing database drop or financial refund tools in the global tool schema relies entirely on the LLM's 'judgment' 
to enforce permissions. An adversary can easily bypass this with prompt injection. Always enforce RBAC filtering dynamically.
""")

st.error("""
**Trap 2: Circular Agent Handoff Ping-Pong Loops**  
If Agent A transfers to Agent B, and Agent B immediately transfers back to Agent A without resolving the user's intent, 
the agents will enter an infinite handoff loop. Always enforce recursion limits on handoffs.
""")

st.warning("""
**Trap 3: Passing Entire Conversation Histories Across Agent Handoff Boundaries**  
Dumping 50 turns of conversational context into a specialist subagent wastes tokens and distracts reasoning. 
Pass a concise handoff briefing containing only relevant facts.
""")

st.warning("""
**Trap 4: Using High-Latency Frontier Models for Basic Triage Routing**  
Using GPT-4o or Sonnet 3.5 merely to decide between 'billing' and 'tech support' introduces an unnecessary 
1,000ms delay into every customer interaction. Use lightweight SLMs.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Agent Handoff</h4>
        <p>A control transfer pattern where an agent invokes a specialized tool to pass conversation thread ownership to another agent.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Dynamic Tool Binding</h4>
        <p>Restricting and configuring the model's available tool schema at runtime based on authenticated user RBAC permissions.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Subagent Isolation</h4>
        <p>Encapsulating intermediate scratchpad state within worker agents to prevent context bloat in the main conversation thread.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Agent Middleware</h4>
        <p>An interception layer executing cross-cutting operational concerns such as PII masking, guardrails, and token throttling.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I understand the architectural difference between central supervisor routing and explicit agent handoffs.", value=True)
chk_2 = st.checkbox("I know how to implement dynamic tool binding based on user Role-Based Access Control (RBAC).", value=True)
chk_3 = st.checkbox("I can isolate subagent state to prevent intermediate observation dumps from bloating main threads.", value=True)
chk_4 = st.checkbox("I know how to assign dynamic models (SLMs vs Frontier) across different StateGraph nodes.", value=True)
chk_5 = st.checkbox("I understand dynamic prompt engineering with conditional directive injection.", value=True)
chk_6 = st.checkbox("I can implement agent middleware for PII sanitization and loop circuit breaking.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Advanced Agent Architecture competencies verified! Ready for Advanced State Management.")
