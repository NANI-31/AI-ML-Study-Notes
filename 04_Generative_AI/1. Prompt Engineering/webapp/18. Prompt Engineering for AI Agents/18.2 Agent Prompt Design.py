import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="18.2 Agent Prompt Design | GenAI Roadmap",
    page_icon="🧠",
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
    .agent-tag {
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
st.title("🧠 18.2 Agent Prompt Design — Role Framing, Planning Protocols & Stop Conditions")
st.markdown(
    """
Welcome to **Lesson 18.2: Agent Prompt Design**.  
Prompting an autonomous agent is fundamentally different from prompting a standard LLM. A standard prompt asks for a single answer; an **agent system prompt establishes the entire operating cognitive architecture** of the system.

A production agent prompt must explicitly define the agent's **role persona**, **objective priorities**, **rules of engagement for tool use**, **step-by-step reasoning protocols**, and **unambiguous stop conditions** to prevent runaway executions.
"""
)

st.divider()

# ==============================================================================
# Section 1: The Canonical 6-Section Agent System Prompt Architecture
# ==============================================================================
st.header("1. The Canonical 6-Section Agent System Prompt Schema", divider="rainbow")
st.markdown("Each section in an agent system prompt establishes critical behavioral guardrails:")

schema_sections_df = pd.DataFrame({
    "Prompt Section": [
        "1. <agent_identity>",
        "2. <primary_objective>",
        "3. <available_tools>",
        "4. <tool_rules_of_engagement>",
        "5. <planning_protocol>",
        "6. <stop_conditions>"
    ],
    "Architectural Purpose": [
        "Defines domain expertise, tone, professional boundaries, and behavioral baseline",
        "Declares the high-level goal and success criteria that define task completion",
        "Lists available tools, their intended capabilities, and input/output contracts",
        "Declares non-negotiable negative constraints (e.g. read-only mode, confirmation gates)",
        "Forces the model to produce explicit 'Thought' reasoning traces before taking actions",
        "Deterministic criteria dictating when the agent must terminate execution"
    ],
    "Prevents What Risk?": [
        "Identity drift and sycophantic persona shifts",
        "Aimless meandering and premature task abandonment",
        "Tool hallucination (invoking non-existent functions)",
        "Destructive actions, unauthorized database mutations, security breaches",
        "Impulsive, uncalibrated tool invocations without a plan",
        "Infinite loops, token budget depletion, runaway cloud bills"
    ]
})
st.dataframe(schema_sections_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Complete Enterprise SRE Agent Blueprint
# ==============================================================================
st.header("2. Complete Production Agent Blueprint: DevOps SRE Agent", divider="rainbow")
st.markdown("A full-scale production system prompt for an autonomous Site Reliability Engineering agent:")

st.code(
    '''<agent_identity>
You are an autonomous Senior Site Reliability Engineering (SRE) Incident Commander.
You operate with methodological precision, absolute factual grounding, and defensive caution.
</agent_identity>

<primary_objective>
Investigate production alerts, isolate root-cause service degradations, and draft an incident remediation brief.
</primary_objective>

<available_tools>
- query_prometheus(query: str, time_range: str): Read cluster CPU, memory, and latency metrics.
- read_kubernetes_logs(pod_name: str, tail_lines: int): Retrieve container stdout/stderr logs.
- describe_pod_status(namespace: str, pod_prefix: str): Inspect deployment crashloop states.
- restart_deployment(deployment_name: str): RESTARTS A CLUSTER SERVICE. (DESTRUCTIVE ACTION).
</available_tools>

<tool_rules_of_engagement>
1. READ-ONLY PRIORITY: Always gather diagnostic evidence using query_prometheus and read_kubernetes_logs first.
2. HUMAN-IN-THE-LOOP (HITL) GATE: You are STRICTLY FORBIDDEN from invoking restart_deployment without explicit
   human confirmation. If restart is needed, draft a proposed recommendation and pause.
3. CONCURRENT AUDITING: When investigating multiple services, query metrics in parallel.
</tool_rules_of_engagement>

<planning_protocol>
Before invoking any tool, you must generate a structured cognitive trace:
- THOUGHT: Explain your reasoning, hypothesis, and why this specific tool is required.
- ACTION: Emit the exact tool call with parameters.
- REFLECTION: After receiving the tool observation, evaluate whether the hypothesis was confirmed or disproved.
</planning_protocol>

<stop_conditions>
Terminate the agent loop and deliver the final incident report when:
1. Root cause is definitively identified and corroborated by logs and metrics, OR
2. A destructive remediation step is required (requiring human manager sign-off), OR
3. Maximum investigation depth of 6 tool turns has been reached without conclusive findings.
Never loop indefinitely.
</stop_conditions>''',
    language="xml",
)

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: The Agent Cognitive Governance Engine", divider="rainbow")
st.markdown("How the agent system prompt governs each step of the perception-action loop:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        rules [label="Agent System Prompt\n(Rules of Engagement)", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        thought [label="Step 1: Thought\n(Reasoning Trace)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        gate [label="HITL Gatekeeper\n(Destructive Check)", fillcolor="#EAB308", fontcolor="#000000"]
        tool [label="Step 2: Safe Tool Call\n(Execute Query)", fillcolor="#3B82F6", fontcolor="#FFFFFF"]
        eval_node [label="Step 3: Stop Check\n(Goal Met or Limit?)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        final_rep [label="Final User Report", fillcolor="#10B981", fontcolor="#FFFFFF"]

        rules -> thought -> gate
        gate -> tool [label=" Read-Only OK", color="#10B981"]
        gate -> final_rep [label=" Requires Human", color="#EF4444"]
        tool -> eval_node
        eval_node -> thought [label=" Iterate", style="dashed"]
        eval_node -> final_rep [label=" Goal Met", color="#10B981"]
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Agent System Prompt Configurator
# ==============================================================================
st.header("4. Interactive Agent System Prompt Configurator", divider="rainbow")
st.markdown("Configure agent behavioral dimensions and generate production-ready prompts:")

agent_domain = st.selectbox(
    "Choose Agent Application Domain",
    [
        "Fintech Compliance & Audit Agent",
        "DevOps SRE Autonomous Triage Agent",
        "Healthcare Clinical Intake Agent"
    ]
)

enforce_hitl = st.checkbox("Enforce Human-in-the-Loop (HITL) Gate on Sensitive Actions", value=True)
require_react = st.checkbox("Enforce ReAct Thought/Action/Reflection Protocol", value=True)
max_steps = st.slider("Maximum Iteration Ceiling (Stop Condition)", min_value=3, max_value=10, value=5)

col_pc1, col_pc2 = st.columns([1, 1])

with col_pc1:
    st.subheader("⚙️ Behavioral Configuration")
    st.markdown(f"- **Domain:** `{agent_domain}`")
    st.markdown(f"- **HITL Gate Active:** `{enforce_hitl}` (Blocks write operations)")
    st.markdown(f"- **Planning Protocol:** `{'ReAct (Thought -> Action -> Reflection)' if require_react else 'Direct Tool Calling'}`")
    st.markdown(f"- **Stop Condition Budget:** `{max_steps} iterations maximum`")

with col_pc2:
    st.subheader("📄 Compiled Agent System Prompt")
    
    compiled_prompt = f"""<agent_identity>
You are an enterprise {agent_domain}.
Maintain professional precision, strict confidentiality, and verifiable factual accuracy.
</agent_identity>

<rules_of_engagement>
1. LEAST PRIVILEGE: Query only data sources relevant to the current user goal.
{'2. HUMAN-IN-THE-LOOP: Destructive actions or financial transactions require human sign-off before execution.' if enforce_hitl else '2. AUTONOMOUS: Full execution authority granted.'}
</rules_of_engagement>

<planning_protocol>
{'Before each action, generate: THOUGHT (hypothesis) -> ACTION (tool call) -> REFLECTION (evaluate observation).' if require_react else 'Invoke tools directly as needed.'}
</planning_protocol>

<stop_conditions>
Halt execution and report final results when:
1. Primary objective is satisfied and verified against evidence.
2. Hard iteration limit of {max_steps} turns is reached.
</stop_conditions>"""

    st.code(compiled_prompt, language="xml")

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Agent Prompt Builder Class", divider="rainbow")
st.markdown("Python builder class to standardize agent prompt construction across microservices:")

st.code(
    '''from typing import List, Optional

class AgentPromptBuilder:
    """
    Constructs production-grade agent system prompts with strict XML encapsulation.
    """
    def __init__(self, role_title: str):
        self.role_title = role_title
        self.objective: Optional[str] = None
        self.tools: List[str] = []
        self.safety_rules: List[str] = []
        self.max_steps: int = 5

    def set_objective(self, objective: str) -> "AgentPromptBuilder":
        self.objective = objective.strip()
        return self

    def add_tool_spec(self, tool_name: str, desc: str) -> "AgentPromptBuilder":
        self.tools.append(f"- {tool_name}: {desc}")
        return self

    def add_safety_rule(self, rule: str) -> "AgentPromptBuilder":
        self.safety_rules.append(rule.strip())
        return self

    def set_max_steps(self, max_steps: int) -> "AgentPromptBuilder":
        self.max_steps = max_steps
        return self

    def compile(self) -> str:
        tools_block = "\\n".join(self.tools)
        rules_block = "\\n".join(f"- {r}" for r in self.safety_rules)
        
        return f"""<agent_identity>
You are an autonomous {self.role_title}.
</agent_identity>

<primary_objective>
{self.objective}
</primary_objective>

<available_tools>
{tools_block}
</available_tools>

<rules_of_engagement>
{rules_block}
</rules_of_engagement>

<stop_conditions>
1. Cease execution and present final answer once primary objective is verified.
2. Cease execution if maximum turn limit of {self.max_steps} iterations is reached.
</stop_conditions>"""

# Example execution:
agent_prompt = (
    AgentPromptBuilder("Financial Forensics Investigator")
    .set_objective("Trace wire fraud transfers over $50k in Q3")
    .add_tool_spec("query_swift_wires", "Searches cross-border SWIFT transaction logs")
    .add_safety_rule("Never modify account balances; read-only access")
    .set_max_steps(4)
    .compile()
)
print(agent_prompt)
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
    with st.expander("Q1: Why is defining explicit Stop Conditions the most critical section of an Agent System Prompt?", expanded=True):
        st.markdown(
            """
Autonomous agents operate inside `while True` perception-action loops. Without unambiguous stop criteria:
1. **Looping Deadlocks**: The agent gets confused by incomplete data and repeatedly queries tools in an endless cycle.
2. **Runaway Cost**: A runaway loop can burn through millions of tokens in minutes.
3. **Enterprise Safeguards**: Clear stop conditions (e.g. *"Stop once hypothesis is verified, stop on error threshold, stop at turn 5"*) guarantee that the agent terminates cleanly with an executive failure summary rather than freezing the application.
"""
        )
    with st.expander("Q2: How do you design an agent prompt to enforce Human-in-the-Loop (HITL) approval gates?"):
        st.markdown(
            """
Include a dedicated **Action Tiering Protocol** in the prompt:
- **Tier 1 (Read-Only / Low Risk)**: The agent has autonomous permission to invoke (e.g. `query_metrics`, `read_logs`).
- **Tier 2 (Destructive / High Risk)**: The agent is forbidden from calling directly (e.g. `delete_database`, `execute_wire_transfer`). The prompt commands the agent to output a structured proposal (`{"action": "REQUEST_HUMAN_APPROVAL", "proposed_tool": "...", "justification": "..."}`) and enter a waiting state until a human operator signs off via webhook.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we just tell the agent: \"Do whatever it takes to solve the user problem\"?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, this gives the agent maximum autonomy and creativity."*
- **Winning Response**: *"Never. Giving an autonomous agent an unbounded objective ('do whatever it takes') without negative rules of engagement is an existential security and availability hazard. An agent tasked with 'fixing server latency' might decide the fastest way to drop latency is to terminate all active user sessions or delete resource-heavy customer data. Production agent prompts must strictly bound the permitted action space with negative rules and least privilege."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Rules of Engagement**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The explicit negative constraints in an agent prompt that define what tools the agent is FORBIDDEN from executing without human approval.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Stop Condition**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Explicit termination criteria that halt an agent loop and force the delivery of a final answer or escalation.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**HITL Approval Gate**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** A checkpoint in an agent workflow that pauses autonomous execution until a human administrator approves a sensitive action.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 18.2 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Architectural Blueprint**: Agent prompts define identity, objectives, tools, rules, planning, and stop conditions.
2. **Defensive Governance**: Enforce Human-in-the-Loop gates on high-risk or destructive tools.
3. **Structured Reasoning**: Require ReAct planning protocols before action execution.
4. **Guaranteed Termination**: Always specify hard step limits and objective completion criteria.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I know the canonical 6-section agent prompt architecture.", value=True, key="chk_18_2_1")
    st.checkbox("I understand why unbounded agent goals create security and cost risks.", value=True, key="chk_18_2_2")
    st.checkbox("I can implement an AgentPromptBuilder class in Python.", value=True, key="chk_18_2_3")
    st.checkbox("I know how to design Human-in-the-Loop approval gates in agent prompts.", value=True, key="chk_18_2_4")
