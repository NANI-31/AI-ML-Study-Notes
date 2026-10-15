import streamlit as st
import json

st.set_page_config(
    page_title="25.1 Prompt Engineering Libraries - Prompt Engineering",
    page_icon="📚",
    layout="wide"
)

# Header Section
st.title("25.1 Frameworks & Libraries: LangChain, LangGraph, LlamaIndex, DSPy, & Semantic Kernel")
st.caption("Chapter 25: Prompt Engineering Frameworks and Ecosystem | Domain: 1. Prompt Engineering")
st.markdown("""
As prompt architectures evolved from raw string manipulation to compound multi-agent workflows, 
a rich ecosystem of frameworks emerged. Choosing the right tool requires understanding whether your problem 
demands **composable prompt chains (LangChain), cyclical state machines (LangGraph), deep index retrieval (LlamaIndex), 
declarative algorithmic prompt compilation (DSPy), or enterprise Microsoft C#/Python integration (Semantic Kernel)**.
""")

st.divider()

# Section 1: The Top 5 Prompt Engineering Libraries
st.header("1. Comparative Analysis of Leading Frameworks", divider="rainbow")
st.markdown("""
Each library addresses a distinct architectural philosophy in modern generative AI development:
""")

col_f1, col_f2 = st.columns(2)
with col_f1:
    st.subheader("1. LangChain (LCEL)")
    st.markdown("""
    * **Philosophy**: Modular Lego blocks for LLMs. Uses LangChain Expression Language (LCEL) for declarative pipe composition (`prompt | model | parser`).
    * **Strengths**: Giant ecosystem of 700+ integrations (vector DBs, LLMs, document loaders).
    * **Best For**: Rapid prototyping, standard RAG pipelines, simple sequential prompt chains.
    """)

    st.subheader("2. LangGraph")
    st.markdown("""
    * **Philosophy**: Cyclical graph-based state machine for multi-agent systems.
    * **Strengths**: Native support for loops, branching decisions, persistent state checkpoints, human-in-the-loop approvals, and multi-agent coordination.
    * **Best For**: Production autonomous agents, complex workflows requiring rollback and memory persistence.
    """)

    st.subheader("3. LlamaIndex")
    st.markdown("""
    * **Philosophy**: Data framework specifically optimized for connecting LLMs to custom data sources.
    * **Strengths**: Superior document chunking, multi-stage retrieval, tree indexes, metadata filters, and knowledge graph querying.
    * **Best For**: Enterprise search, dense RAG, complex unstructured document reasoning.
    """)

with col_f2:
    st.subheader("4. DSPy (Declarative Self-Improving Prompts)")
    st.markdown("""
    * **Philosophy**: **Compile, don't write prompts.** Replaces manual string hacking with typed Signatures and Modules.
    * **Strengths**: Optimizers (e.g. BootstrapFewShot, MIPRO) automatically synthesize, tune, and select optimal few-shot exemplars and system instructions against a validation metric!
    * **Best For**: Rigorous ML engineering, automatic prompt optimization, beating handcrafted prompts on benchmark scores.
    """)

    st.subheader("5. Semantic Kernel (Microsoft)")
    st.markdown("""
    * **Philosophy**: Enterprise-grade orchestration designed for native integration with Microsoft Azure, C#, and Python ecosystems.
    * **Strengths**: Seamless compatibility with Microsoft Copilot plugins, native dependency injection, and security compliance.
    * **Best For**: Enterprise .NET and Azure-native corporate applications.
    """)

st.divider()

# Section 2: Code Comparison Across Frameworks
st.header("2. Framework Code Paradigms Side-by-Side", divider="rainbow")
st.markdown("Examine how the exact same extraction task is expressed across LangChain, DSPy, and LangGraph:")

lib_choice = st.selectbox("Select Framework to Inspect:", ["LangChain (LCEL)", "DSPy (Compiled Optimization)", "LangGraph (State Machine)"])

if lib_choice == "LangChain (LCEL)":
    st.markdown("#### LangChain Expression Language (LCEL) Pipe Syntax")
    st.code('''from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# 1. Define Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract person and company from user input as valid JSON."),
    ("user", "{user_input}")
])

# 2. Compose Pipeline via Unix-style pipe operator (|)
chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | JsonOutputParser()

# 3. Invoke with typed dict
result = chain.invoke({"user_input": "Satya Nadella spoke at Microsoft."})
print(result)  # {"person": "Satya Nadella", "company": "Microsoft"}''', language="python")

elif lib_choice == "DSPy (Compiled Optimization)":
    st.markdown("#### DSPy: Algorithmic Prompt Programming (No Manual Strings)")
    st.code('''import dspy

# 1. Define Typed Signature
class EntityExtractor(dspy.Signature):
    """Extract person and company names from raw text."""
    user_input = dspy.InputField(desc="Raw sentence to analyze")
    person = dspy.OutputField(desc="Full name of person")
    company = dspy.OutputField(desc="Company name")

# 2. Define Module
class ExtractorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(EntityExtractor)

    def forward(self, user_input):
        return self.prog(user_input=user_input)

# 3. DSPy Teleprompter compiles and optimizes the prompt automatically!
# optimizer = dspy.BootstrapFewShot(metric=accuracy_metric)
# compiled_module = optimizer.compile(ExtractorModule(), trainset=train_data)''', language="python")

else:
    st.markdown("#### LangGraph: Cyclical State Graph with Human-in-the-Loop")
    st.code('''from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

# 1. Define Shared Graph State
class AgentState(TypedDict):
    input_text: str
    extracted_entities: dict
    validation_status: str

# 2. Define Node Functions
def extract_node(state: AgentState):
    # Call LLM to extract entities
    return {"extracted_entities": {"person": "Alice", "company": "Acme"}}

def validator_node(state: AgentState):
    entities = state["extracted_entities"]
    is_valid = bool(entities.get("person") and entities.get("company"))
    return {"validation_status": "VALID" if is_valid else "INVALID"}

# 3. Assemble State Graph with Conditional Edge
builder = StateGraph(AgentState)
builder.add_node("extract", extract_node)
builder.add_node("validate", validator_node)

builder.set_entry_point("extract")
builder.add_edge("extract", "validate")
builder.add_conditional_edges(
    "validate",
    lambda state: "end" if state["validation_status"] == "VALID" else "retry",
    {"end": END, "retry": "extract"}
)
graph = builder.compile()''', language="python")

st.divider()

# Section 3: Architecture Decision Matrix
st.header("3. Architecture Decision Matrix", divider="rainbow")
st.markdown("""
When should your team adopt a heavy framework versus remaining on lightweight vanilla SDKs?
""")

st.graphviz_chart('''
digraph FrameworkDecider {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10];
    
    Start [label="New Generative AI Project", fillcolor="#E1BEE7"];
    Q1 [label="Is task simple RAG or\\nsingle-turn prompt?", shape=diamond, fillcolor="#FFF9C4"];
    Vanilla [label="Vanilla OpenAI / Anthropic SDK\\nMinimal dependencies, lowest tech debt", fillcolor="#C8E6C9"];
    
    Q2 [label="Does the system require\\nloops, memory, or agents?", shape=diamond, fillcolor="#FFF9C4"];
    LangGraph [label="LangGraph / CrewAI\\nState machines & multi-agent routing", fillcolor="#BBDEFB"];
    
    Q3 [label="Do you have labeled data\\nto auto-optimize prompts?", shape=diamond, fillcolor="#FFF9C4"];
    DSPy [label="DSPy\\nAlgorithmic prompt compilation", fillcolor="#FFCDD2"];
    LlamaIndex [label="LlamaIndex\\nDense enterprise document search", fillcolor="#FFF9C4"];
    
    Start -> Q1;
    Q1 -> Vanilla [label="Yes (Simple)"];
    Q1 -> Q2 [label="No (Complex)"];
    Q2 -> LangGraph [label="Yes (Agents/State)"];
    Q2 -> Q3 [label="No (Data/Search)"];
    Q3 -> DSPy [label="Yes (Optimize)"];
    Q3 -> LlamaIndex [label="No (Enterprise RAG)"];
}
''', use_container_width=False)

st.divider()

# Section 4: Interactive Framework Selector Workbench
st.header("4. Interactive Framework Decision Workbench", divider="rainbow")
st.markdown("Answer 3 architecture questions to receive an objective framework recommendation.")

col_q1, col_q2, col_q3 = st.columns(3)
with col_q1:
    req_type = st.selectbox("Primary Architecture Focus:", ["Complex Multi-Step Agents with State", "High-Precision Prompt Optimization", "Enterprise Document RAG", "Simple 2-Step Extraction"])
with col_q2:
    team_skill = st.selectbox("Team Preference:", ["Python Data Science", "Enterprise C# / .NET", "Lean Minimalist (Zero Dependency)"])
with col_q3:
    eval_data = st.selectbox("Do you have 50+ labeled examples?", ["Yes, golden dataset available", "No, pure zero-shot prototyping"])

if req_type == "High-Precision Prompt Optimization" and "Yes" in eval_data:
    st.success("🎯 **Recommended Framework: DSPy**")
    st.markdown("Since you have labeled data and prioritize precision, DSPy's teleprompters will automatically beat handcrafted prompts by discovering the optimal few-shot exemplars.")
elif req_type == "Complex Multi-Step Agents with State":
    st.success("🎯 **Recommended Framework: LangGraph**")
    st.markdown("LangGraph's state machine handles cycles, persistent checkpointing, and human approval gates better than any pipeline library.")
elif req_type == "Enterprise Document RAG":
    st.success("🎯 **Recommended Framework: LlamaIndex**")
    st.markdown("LlamaIndex offers the most advanced chunking, reranking, and hierarchical indexing primitives for enterprise document reasoning.")
elif team_skill == "Enterprise C# / .NET":
    st.success("🎯 **Recommended Framework: Semantic Kernel**")
    st.markdown("Semantic Kernel is purpose-built for Microsoft ecosystems, Azure OpenAI integrations, and C# developer workflows.")
else:
    st.info("💡 **Recommended Approach: Vanilla Provider SDK (OpenAI / Anthropic)**")
    st.markdown("Avoid framework abstraction overhead! Direct SDK calls paired with Pydantic provide maximum transparency and zero dependency rot.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa1, qa2 = st.columns(2)

with qa1:
    with st.expander("Q: What is the core paradigm shift introduced by DSPy?"):
        st.write("""
        DSPy treats prompt engineering like machine learning compilation rather than manual copywriting. 
        Instead of tweaking adjectives in prompt strings, developers declare typed inputs/outputs (Signatures) 
        and modular pipelines. DSPy teleprompters (optimizers) then search the space of possible few-shot exemplars 
        and instruction phrasings, compiling the prompt that maximizes an objective evaluation metric over a training dataset.
        """)

    with st.expander("Q: Why did LangGraph emerge to replace classic LangChain AgentExecutor?"):
        st.write("""
        Classic LangChain `AgentExecutor` was a rigid, black-box linear loop that struggled with cyclical workflows, 
        state persistence, human-in-the-loop approvals, and multi-agent collaboration. 
        LangGraph models agent systems as explicit Directed Graphs with typed state and customizable edges, 
        giving developers complete control over decision routing and state rollback.
        """)

with qa2:
    with st.expander("Trap to Avoid: Framework Lock-in Overhead"):
        st.write("""
        Adopting heavyweight orchestration frameworks prematurely often leads to debugging cryptic abstractions, 
        breaking changes across minor version upgrades, and hidden token bloat. 
        Always start with direct provider SDKs and Pydantic. Only adopt LangGraph or LlamaIndex when your system 
        genuinely requires cyclical state graphs or complex multi-document indexing that would be arduous to build from scratch.
        """)

    with st.expander("Flashcards: Framework Ecosystem"):
        fc_sel = st.selectbox(
            "Select Framework Term:",
            ["LCEL (LangChain Expression Language)", "DSPy Teleprompter", "LangGraph StateGraph", "LlamaIndex RouterQueryEngine"]
        )
        if fc_sel == "LCEL (LangChain Expression Language)":
            st.info("A declarative syntax using the pipe operator (|) to compose prompts, models, and parsers into chains.")
        elif fc_sel == "DSPy Teleprompter":
            st.info("An automated prompt optimizer that discovers and compiles the best few-shot exemplars against an eval metric.")
        elif fc_sel == "LangGraph StateGraph":
            st.info("A cyclical state machine framework where nodes represent actions and edges represent decision transitions.")
        else:
            st.info("An engine that dynamically routes user queries to the best specialized vector index or summary tool.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the architectural differences between LangChain, LangGraph, and DSPy.")
    st.checkbox("I know when to choose vanilla SDKs over third-party framework abstractions.")
with c2:
    st.checkbox("I can implement an LCEL prompt chain using pipe syntax.")
    st.checkbox("I understand how DSPy compiles prompts automatically against golden evaluation sets.")
