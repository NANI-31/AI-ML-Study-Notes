import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="4.3 Chat Prompt Templates - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("4.3 Chat Prompt Templates: Multi-Role Composition & MessagesPlaceholder")
st.caption("Chapter 4: Prompt Engineering with LangChain | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
While `PromptTemplate` produces a flat text string, modern chat models require an array of **structured, 
role-annotated messages**. Building multi-turn conversational agents, copilots, and interactive assistants 
demands a templating abstraction that manages system steering, few-shot assistant turns, dynamic user queries, 
and sliding-window conversation history buffers.

**`ChatPromptTemplate`** is the standard multi-role prompt primitive in LangChain (v0.3+). It allows developers 
to compose heterogeneous sequences of `SystemMessagePromptTemplate`, `HumanMessagePromptTemplate`, 
`AIMessagePromptTemplate`, and dynamic `MessagesPlaceholder` containers into a single composable `Runnable`.

In this lesson, you will master:
- **`ChatPromptTemplate` Architecture**: Declarative role-tuple construction via `.from_messages()`.
- **Role-Specific Templates**: Templating dynamic parameters inside system, human, and synthetic AI turns.
- **Dynamic Context Injection**: Using `MessagesPlaceholder` to dynamically inject arbitrary conversation histories.
- **Multi-Turn State Management**: Integrating session memory buffers into LCEL pipelines.
- **Fail-Safe History Handling**: Gracefully handling empty histories without formatting crashes.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockHistoryMessage:
    """Represents a conversational message in module scope."""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class MockChatPromptCompiler:
    """Simulates ChatPromptTemplate compilation with role templates and MessagesPlaceholder."""
    def __init__(self, system_template: str, human_template: str):
        self.system_template = system_template
        self.human_template = human_template

    def compile(self, system_vars: Dict[str, str], human_vars: Dict[str, str], 
                chat_history: List[MockHistoryMessage]) -> List[Dict[str, str]]:
        messages = []
        
        # 1. Format System Message
        sys_text = self.system_template
        for k, v in system_vars.items():
            sys_text = sys_text.replace(f"{{{k}}}", v)
        messages.append({"role": "SystemMessage", "content": sys_text})

        # 2. Inject MessagesPlaceholder (chat history)
        for h in chat_history:
            messages.append({"role": h.role, "content": h.content})

        # 3. Format Human Message
        human_text = self.human_template
        for k, v in human_vars.items():
            human_text = human_text.replace(f"{{{k}}}", v)
        messages.append({"role": "HumanMessage", "content": human_text})

        return messages

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: ChatPromptTemplate Schema & Conversational Context Flow", divider="rainbow")

st.markdown("""
The diagram below illustrates how `ChatPromptTemplate` compiles role-specific message templates 
and dynamic history placeholders into a strongly-typed `ChatPromptValue` sequence ready for model consumption.
""")

dot_blueprint = """
digraph ChatPromptTemplateArchitecture {
    rankdir=TB;
    bgcolor="#0B1120";
    fontname="Helvetica";
    compound=true;

    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    subgraph cluster_declarative {
        label="1. Declarative Message Template Definition";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_decl_sys [label="('system', 'You are an SRE specialized in {domain}')", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_decl_hist [label="MessagesPlaceholder(variable_name='chat_history')", fillcolor="#1E293B", fontcolor="#F59E0B"];
        node_decl_fewshot [label="('ai', 'Acknowledged. Ready to review telemetry.')", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_decl_human [label="('human', '{user_query}')", fillcolor="#1E293B", fontcolor="#F8FAFC"];

        node_decl_sys -> node_decl_hist -> node_decl_fewshot -> node_decl_human [color="#38BDF8"];
    }

    subgraph cluster_compilation {
        label="2. Runtime Variable Interpolation & History Splice";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_inputs [label="Runtime Input Payload:\n- domain: 'Kubernetes Networking'\n- chat_history: [HumanMessage, AIMessage]\n- user_query: 'Check ingress gateway error rate'", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_compiler [label="ChatPromptTemplate.invoke(inputs)\nSplices history & substitutes braces", fillcolor="#1E293B", fontcolor="#38BDF8"];

        node_inputs -> node_compiler;
    }

    subgraph cluster_output {
        label="3. Concrete Message Sequence (ChatPromptValue)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_out_seq [label="List[BaseMessage]:\n1. SystemMessage(content='You are an SRE...')\n2. HumanMessage(content='Previous turn...')\n3. AIMessage(content='Previous answer...')\n4. HumanMessage(content='Check ingress...')", fillcolor="#1E293B", fontcolor="#34D399"];

        node_compiler -> node_out_seq [label="produces", color="#34D399"];
    }

    node_decl_human -> node_compiler [style="dashed", color="#38BDF8"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

st.markdown("""
> **Key Architecture Insight**: When `ChatPromptTemplate` is invoked, it returns a `ChatPromptValue`. 
> Unlike `StringPromptValue`, `ChatPromptValue` preserves the list structure of concrete `BaseMessage` objects, 
> allowing Chat Models to cleanly map each message into provider-specific API payloads.
""")

# --- SECTION 2: CHATPROMPTTEMPLATE ---
st.header("2. ChatPromptTemplate: Construction, Tuple Syntax & Multi-Turn Composition", divider="rainbow")

st.markdown("""
In LangChain v0.3+, the canonical method to construct a `ChatPromptTemplate` is using **role tuples** 
via `ChatPromptTemplate.from_messages()`. 

A role tuple has the signature `("role", "template_string")`, where the role can be `"system"`, `"human"`, `"ai"`, 
or `"user"`, `"assistant"`.
""")

st.code('''from langchain_core.prompts import ChatPromptTemplate

# Canonical role-tuple syntax
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an enterprise architect assisting with {platform} migrations."),
    ("human", "Audit configuration file: {config_file_name}\\n\\n{content}")
])

# Inspecting automatically extracted variables
print(prompt.input_variables)  # ['platform', 'config_file_name', 'content']

# Invocations produce a ChatPromptValue
prompt_val = prompt.invoke({
    "platform": "AWS EKS",
    "config_file_name": "ingress.yaml",
    "content": "apiVersion: networking.k8s.io/v1..."
})
print(prompt_val.to_messages())
''', language="python")

# --- SECTION 3: SYSTEM MESSAGES ---
st.header("3. System Messages: Dynamic System Prompts, Templated Directives & Security Guardrails", divider="rainbow")

st.markdown("""
In conversational systems, system prompts are rarely static. They frequently inject dynamic parameters 
such as user authorization levels, client company guidelines, or temporal operational state.
""")

col_sm1, col_sm2 = st.columns(2)
with col_sm1:
    st.markdown("""
    #### Architectural Best Practices
    - **Inject Authority & Security**: Place non-negotiable compliance rules in the system prompt.
    - **Parametrize Tenant Isolation**: Inject tenant IDs and authorization scopes dynamically.
    - **Provider Adaptation**: LangChain automatically adapts system messages for models that require system instructions in a separate top-level parameter (e.g., Anthropic Claude).
    """)
with col_sm2:
    st.code('''from langchain_core.prompts import ChatPromptTemplate

templated_system_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a HIPAA Compliance Auditor.
Auditing entity: {client_name}
Enforcement level: {audit_tier}
Never disclose Protected Health Information (PHI)."""),
    ("human", "{audit_query}")
])

messages = templated_system_prompt.invoke({
    "client_name": "St. Jude Medical Group",
    "audit_tier": "STRICT_AUDIT",
    "audit_query": "Verify patient records encryption algorithm."
})
''', language="python")

# --- SECTION 4: HUMAN MESSAGES ---
st.header("4. Human Messages: Templated User Input Framing & Multimodal Block Interpolation", divider="rainbow")

st.markdown("""
A `HumanMessagePromptTemplate` defines the template structure for user turns. 
In addition to simple text interpolation, modern templates can dynamically construct **multimodal payloads** 
containing image URLs and base64 strings.
""")

st.code('''from langchain_core.prompts import ChatPromptTemplate

# Multi-variable HumanMessage framing
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a DevOps code reviewer."),
    ("human", """Review git diff for service '{service_name}':
Branch: {branch_name}
Diff:
{diff_content}""")
])

# Can also template multimodal image blocks:
# ("human", [{"type": "text", "text": "{instructions}"}, {"type": "image_url", "image_url": "{image_url}"}])
''', language="python")

# --- SECTION 5: AI MESSAGES ---
st.header("5. AI Messages: Few-Shot Exemplars, Synthetic Assistant Turns & Priming Reasoning", divider="rainbow")

st.markdown("""
Including synthetic `AIMessage` turns inside a `ChatPromptTemplate` is a powerful technique for 
**In-Context Priming**. By providing a fake prior assistant turn, you demonstrate the exact reasoning format, 
tone, and preliminary acknowledgement you expect from the model.
""")

st.code('''from langchain_core.prompts import ChatPromptTemplate

primed_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an automated incident triager."),
    ("human", "Alert: P99 latency on checkout service exceeded 1200ms."),
    # Synthetic AI exemplar turn:
    ("ai", "Acknowledged. Initiating triage workflow: 1) Checking database connection pools, 2) Inspecting Redis cache hit ratios."),
    ("human", "Database connection pool is at 98% utilization. What is the immediate mitigation?")
])

# The model continues the conversation having observed the structured AI exemplar!
''', language="python")

# --- SECTION 6: PLACEHOLDER MESSAGES ---
st.header("6. Placeholder Messages: MessagesPlaceholder Dynamic Injection & Optional Lists", divider="rainbow")

st.markdown("""
In real-world chat applications, you do not know in advance how many conversational turns the user has taken. 
You cannot hardcode a fixed number of `("human", ...)` and `("ai", ...)` tuples.

**`MessagesPlaceholder`** acts as a dynamic container that splices an arbitrary list of `BaseMessage` objects 
directly into that slot during invocation.
""")

dot_placeholder_flow = """
digraph MessagesPlaceholderFlow {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    node_template [label="ChatPromptTemplate Structure:\n[SystemMessage, MessagesPlaceholder('chat_history'), HumanMessage]", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    node_history_list [label="chat_history variable:\n[HumanMessage('Turn 1'), AIMessage('Answer 1'), HumanMessage('Turn 2')]", fillcolor="#1E293B", fontcolor="#F59E0B"];
    node_assembled [label="Assembled Result:\n[SystemMessage, HumanMessage('Turn 1'), AIMessage('Answer 1'), HumanMessage('Turn 2'), HumanMessage('Current Turn')]", fillcolor="#1E293B", fontcolor="#34D399"];

    node_template -> node_assembled [label="compiles with", color="#38BDF8"];
    node_history_list -> node_assembled [label="spliced into placeholder slot", color="#F59E0B"];
}
"""

with st.expander("🔍 Contextual Diagram: MessagesPlaceholder Dynamic Splicing", expanded=True):
    st.graphviz_chart(dot_placeholder_flow, use_container_width=True)

st.code('''from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Defining a template with MessagesPlaceholder
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer support engineer."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),  # optional=True allows empty list
    ("human", "{user_query}")
])

# Invoking with conversation history
history = [
    HumanMessage(content="My cluster node-03 went down."),
    AIMessage(content="I see node-03 failed health checks due to out-of-memory errors.")
]

compiled_value = chat_prompt.invoke({
    "chat_history": history,
    "user_query": "How do I increase the memory allocation limit?"
})
''', language="python")

# --- SECTION 7: MESSAGE HISTORY ---
st.header("7. Message History: Stateful Memory Integration, Thread Session Binding & Sliding Windows", divider="rainbow")

st.markdown("""
Integrating `ChatPromptTemplate` with external databases (e.g., Redis, DynamoDB, or PostgreSQL) 
allows your system to persist conversational state across distinct HTTP requests.

LangChain provides **`RunnableWithMessageHistory`** to automatically fetch, inject, and persist 
conversation histories based on a `session_id`.
""")

st.code('''from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# 1. Base Prompt and Model
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding copilot."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
model = ChatOpenAI(model="gpt-4o", temperature=0.1)
chain = prompt | model

# 2. In-memory session store (Redis in production)
session_store = {}

def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# 3. Stateful Runnable wrapper
stateful_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Invocations automatically persist state per session_id:
# res1 = stateful_chain.invoke({"input": "My favorite language is Rust"}, config={"configurable": {"session_id": "user_123"}})
# res2 = stateful_chain.invoke({"input": "What is my favorite language?"}, config={"configurable": {"session_id": "user_123"}})
''', language="python")

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Enterprise Multi-Turn Chat Prompt Pipeline with Memory", divider="rainbow")

st.markdown("""
Below is an enterprise-grade conversational pipeline integrating `ChatPromptTemplate`, `MessagesPlaceholder`, 
and automated sliding-window message trimming.
""")

st.code('''from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_openai import ChatOpenAI

class EnterpriseChatPipeline:
    """Production chat pipeline with parametrized system directives and token-bounded history."""

    def __init__(self, api_key: str):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=api_key)
        
        # Assemble declarative ChatPromptTemplate
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Site Reliability Engineer.
Platform: {infrastructure_platform}
Environment: {env_name}
Directive: Diagnose alerts using first principles. Keep answers concise."""),
            MessagesPlaceholder(variable_name="conversation_history", optional=True),
            ("human", "{current_query}")
        ])

    def execute_turn(
        self,
        platform: str,
        env: str,
        history: List[BaseMessage],
        query: str
    ) -> str:
        # Bounding history to prevent context overflow
        bounded_history = trim_messages(
            history,
            max_tokens=800,
            strategy="last",
            token_counter=self.model,
            include_system=False
        )

        # Assemble and execute LCEL pipeline
        pipeline = self.prompt_template | self.model
        ai_response: AIMessage = pipeline.invoke({
            "infrastructure_platform": platform,
            "env_name": env,
            "conversation_history": bounded_history,
            "current_query": query
        })

        return str(ai_response.content)
''', language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Interactive ChatPromptTemplate Compiler & History Injector", divider="rainbow")

st.markdown("""
Experiment with the live `ChatPromptTemplate` compiler below. Adjust **System variables**, 
add multi-turn conversation messages into the **MessagesPlaceholder**, and inspect the 
concrete message array generated for the model.
""")

col_wb1, col_wb2 = st.columns(2)
with col_wb1:
    sys_template_wb = st.text_area(
        "System Message Template",
        value="You are a Principal Architect for {cloud_provider} environments.\nCompliance: {compliance_tier}"
    )
    cloud_var = st.text_input("Var 'cloud_provider'", value="AWS Cloud")
    compliance_var = st.text_input("Var 'compliance_tier'", value="SOC2-Compliant")

with col_wb2:
    human_template_wb = st.text_area(
        "Human Message Template",
        value="Target Host: {target_host}\nQuestion: {query}"
    )
    host_var = st.text_input("Var 'target_host'", value="bastion.prod.internal")
    query_var = st.text_input("Var 'query'", value="Check for unauthorized SSH logins.")

st.subheader("💬 Spliced Chat History (MessagesPlaceholder)")
if "chat_history_state" not in st.session_state:
    st.session_state.chat_history_state = [
        MockHistoryMessage("HumanMessage", "Initiating morning audit."),
        MockHistoryMessage("AIMessage", "Morning audit initialized. Ready for target inspection.")
    ]

# Display existing history
for i, m in enumerate(st.session_state.chat_history_state):
    st.caption(f"Turn {i+1} ({m.role}): {m.content}")

c_add1, c_add2 = st.columns(2)
with c_add1:
    new_user_msg = st.text_input("Add Mock User Turn to History", value="Previous incident INC-102 was resolved.")
    if st.button("➕ Append to History"):
        st.session_state.chat_history_state.append(MockHistoryMessage("HumanMessage", new_user_msg))
        st.session_state.chat_history_state.append(MockHistoryMessage("AIMessage", f"Acknowledged: {new_user_msg}"))
        st.rerun()

with c_add2:
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history_state = []
        st.rerun()

if st.button("🚀 Compile ChatPromptTemplate", type="primary"):
    compiler = MockChatPromptCompiler(sys_template_wb, human_template_wb)
    compiled_messages = compiler.compile(
        system_vars={"cloud_provider": cloud_var, "compliance_tier": compliance_var},
        human_vars={"target_host": host_var, "query": query_var},
        chat_history=st.session_state.chat_history_state
    )

    st.subheader("📊 Compiled Message Sequence (Ready for Model API)")
    st.info(f"Total Messages in Array: {len(compiled_messages)}")
    for idx, msg in enumerate(compiled_messages):
        badge = "🔵" if msg["role"] == "SystemMessage" else ("🟢" if msg["role"] == "HumanMessage" else "🟣")
        with st.expander(f"{badge} Message {idx + 1}: {msg['role']}", expanded=True):
            st.code(msg["content"], language="markdown")

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("What is the primary difference between PromptTemplate and ChatPromptTemplate?",
     "PromptTemplate produces a single flat formatted string (StringPromptValue) designed for completion models. ChatPromptTemplate produces a structured sequence of BaseMessage objects (ChatPromptValue) where each message possesses a distinct role (system, human, ai, tool), enabling multi-turn conversation modeling and role-specific formatting."),
    ("Why is MessagesPlaceholder essential for multi-turn conversational agents?",
     "In real-world chat applications, conversation length is dynamic. MessagesPlaceholder provides a named slot inside ChatPromptTemplate that accepts an arbitrary list of BaseMessage objects at runtime. Without MessagesPlaceholder, developers would have to rebuild the prompt template string on every conversational turn."),
    ("What happens if you omit optional=True on a MessagesPlaceholder and pass an empty list or omit the variable?",
     "In older versions of LangChain, omitting the variable resulted in a KeyError during template compilation. Setting optional=True (or passing an empty list []) informs the template compiler that the history can be legitimately empty, preventing formatting crashes on the very first turn of a conversation."),
    ("How does ChatPromptTemplate adapt system messages for providers like Anthropic Claude?",
     "While OpenAI accepts system messages as an item within the messages array, Anthropic's Messages API requires system instructions to be passed as a top-level 'system' parameter. LangChain's ChatAnthropic implementation automatically inspects the ChatPromptValue, extracts the SystemMessage, and maps it to the top-level API parameter without developer intervention."),
    ("Why should conversation history be trimmed before being passed into MessagesPlaceholder?",
     "As conversations progress, token count grows monotonically (O(T) complexity). If history is not bounded by a sliding window (e.g., using trim_messages), the prompt will eventually exceed the model's context window limit, leading to API crashes or ballooning inference costs.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Passing Plain Strings into MessagesPlaceholder. Supplying a string like `chat_history='Hello'` instead of a list of `BaseMessage` objects raises a `TypeError` during invocation. Always pass `List[BaseMessage]`.")
    st.error("**Trap 2 (Breaking Error)**: Missing Variables Inside Sub-Templates. Defining `('human', '{user_query} on {target}')` and failing to provide `target` in the input dictionary raises an immediate `KeyError` before the model call.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Unbounded MessagesPlaceholder Injection. Appending every single user turn to an in-memory list without token limits causes rapid token exhaustion. Always pair `MessagesPlaceholder` with `trim_messages`.")
    st.warning("**Trap 4 (Ordering Warning)**: Misplacing MessagesPlaceholder After Current Query. Placing `MessagesPlaceholder` below the current user query reverses chronological flow, confusing the model's attention mechanism. Always place history before current turn.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: ChatPromptTemplate Factory"):
        st.markdown("**ChatPromptTemplate.from_messages([('system', '...'), ('human', '...')])**")
    with st.expander("🎴 Flashcard 2: Dynamic History Slot"):
        st.markdown("**MessagesPlaceholder(variable_name='history', optional=True) injects dynamic message lists.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: ChatPromptValue Output"):
        st.markdown("**Returns ChatPromptValue containing concrete BaseMessage objects.**")
    with st.expander("🎴 Flashcard 4: Stateful Memory Wrapper"):
        st.markdown("**RunnableWithMessageHistory wraps LCEL chains to manage session-based message persistence.**")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can construct ChatPromptTemplates using role-tuple syntax.", value=True)
chk2 = st.checkbox("I understand the difference between StringPromptValue and ChatPromptValue.", value=True)
chk3 = st.checkbox("I know how to inject dynamic conversation history using MessagesPlaceholder.", value=True)
chk4 = st.checkbox("I understand why MessagesPlaceholder should use optional=True.", value=True)
chk5 = st.checkbox("I know how to integrate RunnableWithMessageHistory for session state management.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 4.3 Chat Prompt Templates. Ready for 4.4 Advanced Prompting!")
