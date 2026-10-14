import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="32.3 Agent Memory - Embeddings Curriculum",
    page_icon="🤖",
    layout="wide"
)

# Title & Overview
st.title("32.3 Agent Memory: Interaction Logs, Cognitive Recall, Reflection & Strategic Forgetting")
st.caption("Chapter 32: Embeddings with LLM Applications | Domain: 3. Vector Embeddings")
st.markdown("""
In autonomous AI agent architectures (e.g., LangGraph, AutoGen, CrewAI, MemGPT / Letta), 
an agent does not simply generate single responses—it **plans, invokes external tools, observes environment feedback, 
and executes multi-step workflows across hours or days**.

Without persistent memory, an autonomous agent suffers from severe cognitive failure modes:
* **Repeating Past Failures**: If an API call fails with a 429 Rate Limit error, an ungrounded agent will blindly repeat the exact same call in a loop.
* **Context Overflow**: Multi-step tool outputs (e.g., 50-page SQL dumps or bash terminal outputs) rapidly explode the context window.
* **Lack of Learning**: An agent that solved a complex Kubernetes debugging task yesterday has no recollection of the fix today.

**Embedding-Based Agent Memory** equips autonomous agents with **Episodic Interaction Buffers**, **Pre-Action Cognitive Recall**, 
**Hierarchical Reflection Synthesis**, and **Strategic Forgetting Policies**, transforming fragile script-runners into 
resilient, self-improving autonomous systems.
""")

st.divider()

# Section 1: The 4-Stage Agent Memory Lifecycle
st.header("1. The 4-Stage Autonomous Agent Memory Lifecycle", divider="rainbow")

col_life1, col_life2 = col_life1, col_life2 = st.columns([1, 1])

with col_life1:
    st.markdown("""
    ### The 4 Stages of Agent Memory
    1. **Store Interactions (Episodic Action-Observation Traces)**:
       * When an agent executes an action, it records the structured tuple:
         `(Goal, Plan, Tool_Called, Tool_Output, Outcome_Success_or_Failure)`.
       * The trace is embedded and stored in an episodic vector memory collection.
    2. **Cognitive Recall (Pre-Planning Memory Retrieval)**:
       * Before generating a new execution plan or selecting a tool, the agent queries its past episodic memory:
         * *"Have I attempted this task before? What errors occurred? What tools worked?"*
       * Relevant past experiences are injected into the agent's planning prompt.
    3. **Memory Consolidation & Reflection (Reflection Tree)**:
       * Periodically, the agent reflects on raw observation logs to synthesize higher-level abstract insights:
         * *Raw Log*: *"Calling API /v1/users returned 429 three times in 10 seconds."*
         * *Consolidated Reflection*: *"Rule: API /v1/users has a strict 1-req/sec rate limit; always apply exponential backoff."*
    4. **Strategic Forgetting & Garbage Collection**:
       * Pruning ephemeral, voluminous intermediate tool outputs (e.g., massive stdout logs) while retaining the consolidated reflection.
    """)

with col_life2:
    st.markdown("### Autonomous Agent Memory Architecture")
    agent_mem_dot = """
    digraph AgentMemoryArchitecture {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        AgentTask [label="User Task / Goal\\n'Deploy Service X to AWS ECS'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        CognitiveRecall [label="Stage 1: Pre-Action Recall\\nQueries Vector Memory for Past Experiences\\n'What worked/failed previously?'", fillcolor="#FEF3C7", color="#F59E0B"];

        MemoryStore [label="Episodic Vector Memory\\n(Past Trials, Errors & Insights)\\nChroma / Qdrant", fillcolor="#D1FAE5", color="#10B981", style="filled,bold"];

        Planner [label="Agent Reasoning & Planner\\nFormulates Safe Action Steps", fillcolor="#FDE68A"];

        ToolExec [label="Tool Execution & Environment Action\\n(Run Bash, Query DB, Invoke API)", fillcolor="#DBEAFE"];

        ObsTrace [label="Stage 2: Store Interaction Trace\\nAction + Tool Output + Result", fillcolor="#BFDBFE"];

        ReflectionEngine [label="Stage 3: Reflection & Consolidation\\nExtracts Abstract Learned Rules", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        PruneEngine [label="Stage 4: Strategic Forgetting\\nPurges raw logs; keeps core rules", fillcolor="#FCE7F3", color="#DB2777"];

        AgentTask -> CognitiveRecall;
        MemoryStore -> CognitiveRecall [label="Retrieve Prior Fixes"];
        CognitiveRecall -> Planner;
        Planner -> ToolExec;
        ToolExec -> ObsTrace;
        ObsTrace -> MemoryStore [label="Store raw trace"];
        ObsTrace -> ReflectionEngine;
        ReflectionEngine -> MemoryStore [label="Store high-level rules"];
        MemoryStore -> PruneEngine;
    }
    """
    st.graphviz_chart(agent_mem_dot, use_container_width=False)

st.divider()

# Section 2: Reflection Synthesis & Strategic Forgetting
st.header("2. Reflection Synthesis & Strategic Forgetting Policies", divider="rainbow")

col_ref1, col_ref2 = col_ref1, col_ref2 = st.columns([1, 1])

with col_ref1:
    st.markdown("""
    ### The Reflection Synthesis Mechanism
    In raw execution traces, 95% of text is repetitive terminal output or JSON payloads.
    If an agent keeps all raw logs in its vector store, memory search becomes polluted with low-level noise.
    
    **The Reflection Tree (Generative Reflection)**:
    1. Every $N$ tool interactions (or upon task completion), an LLM reflection agent analyzes the recent episodic trace:
       * *"What worked? What failed? What invariant rule should be remembered for the future?"*
    2. The LLM distills the raw trace into an **Atomic Behavioral Guideline**:
       * *Insight*: *"When deploying Docker images to ECR, the AWS CLI login command must be executed with region eu-west-1 before docker push."*
    3. The reflection is embedded with a high **Importance Rating (9/10)**, making it permanently retrievable across all future agent missions!
    """)

with col_ref2:
    st.markdown("""
    ### Strategic Forgetting & Eviction Policies
    Unchecked memory expansion causes retrieval latency to degrade and bloats database storage costs.
    
    **Production Eviction Policies**:
    * **Time-To-Live (TTL) Expiration**:
      * Ephemeral tool outputs (e.g., temporary session tokens, raw ping outputs) expire after 24 hours.
    * **LRU (Least Recently Used) Memory Eviction**:
      * If memory exceeds storage quotas (e.g., 50,000 vectors), drop the vectors with the lowest access frequency and lowest importance score.
    * **Selective Compression**:
      * Replace 10 detailed raw trial logs with a single consolidated reflection summary, and delete the 10 raw children logs.
    """)

st.divider()

# Section 3: Production Code: The Self-Improving Agent Memory Manager
st.header("3. Production Code: The Self-Improving Agent Memory Manager", divider="rainbow")

st.code("""
import numpy as np
from openai import OpenAI
import chromadb
import time

client = OpenAI()
chroma_client = chromadb.Client()
agent_mem_col = chroma_client.get_or_create_collection(
    name="agent_episodic_memory",
    metadata={"hnsw:space": "cosine"}
)

def get_embedding(text: str) -> list[float]:
    res = client.embeddings.create(model="text-embedding-3-small", input=text)
    return res.data[0].embedding

class AgentMemoryManager:
    def store_interaction(self, goal: str, tool_name: str, outcome: str, success: bool):
        \"\"\"Stores action-observation trace in episodic memory.\"\"\"
        content = f"Goal: {goal} | Tool: {tool_name} | Outcome: {outcome} | Success: {success}"
        mem_id = f"trace_{int(time.time()*1000)}"
        
        agent_mem_col.add(
            ids=[mem_id],
            embeddings=[get_embedding(content)],
            documents=[content],
            metadatas=[{
                "goal": goal,
                "tool": tool_name,
                "success": success,
                "timestamp": time.time(),
                "type": "episodic_trace"
            }]
        )

    def recall_relevant_experience(self, current_goal: str, top_k: int = 3) -> list[str]:
        \"\"\"Pre-planning cognitive recall of past trials.\"\"\"
        q_vec = get_embedding(current_goal)
        results = agent_mem_col.query(query_embeddings=[q_vec], n_results=top_k)
        if results['documents'] and len(results['documents'][0]) > 0:
            return results['documents'][0]
        return []

    def consolidate_reflection(self, recent_traces: list[str]) -> str:
        \"\"\"Synthesizes golden behavioral rule and stores high-importance reflection.\"\"\"
        prompt = f\"\"\"Analyze these recent tool execution traces:
{recent_traces}

Synthesize 1 concise, actionable invariant rule that the agent must remember for future similar tasks.
Rule:\"\"\"
        res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
        rule_text = res.choices[0].message.content.strip()
        
        # Store high-priority reflection
        ref_id = f"rule_{int(time.time())}"
        agent_mem_col.add(
            ids=[ref_id],
            embeddings=[get_embedding(rule_text)],
            documents=[rule_text],
            metadatas=[{"type": "consolidated_rule", "importance": 10, "timestamp": time.time()}]
        )
        return rule_text
""", language="python")

st.divider()

# Section 4: Interactive Workbench — Autonomous Agent Memory Stepper
st.header("4. Interactive Workbench: Autonomous Agent Memory Stepper", divider="rainbow")

st.markdown("""
Step through the complete **4-stage Agent Memory Cycle** live. 
Select an agent mission, trigger **Pre-Planning Cognitive Recall** to inspect past mistakes, execute an action, 
watch the **Reflection Engine** synthesize a golden rule, and observe **Strategic Forgetting** purge temporary logs!
""")

agent_scenarios = {
    "Mission 1: Deploy Microservice to AWS ECS": {
        "goal": "Deploy Docker container billing-v2 to AWS ECS cluster in eu-west-1",
        "past_memories": [
            {"id": "TRACE_01", "text": "Attempted docker push to AWS ECR; failed with 'no basic auth credentials'. Resolved by running 'aws ecr get-login-password' first.", "type": "Past Failure Fix", "importance": 9},
            {"id": "TRACE_02", "text": "ECS Task definition requires minimum 512MB memory limit; 256MB triggers container memory allocation error.", "type": "Past Hardware Limit", "importance": 8},
            {"id": "RAW_LOG_01", "text": "Raw Docker build stdout: Step 1/12 FROM alpine:latest ... 45 lines of build progress logs.", "type": "Ephemeral Log", "importance": 2}
        ],
        "new_reflection": "Rule: Before running 'docker push' to ECR, always authenticate via 'aws ecr get-login-password --region eu-west-1' and allocate >= 512MB in the task definition."
    },
    "Mission 2: Diagnose Postgres Deadlock in Production": {
        "goal": "Investigate and remediate recurring 40P01 deadlock alerts on orders table",
        "past_memories": [
            {"id": "TRACE_11", "text": "Deadlock caused by concurrent transactions acquiring locks in reverse order on table accounts and table orders.", "type": "Root Cause Analysis", "importance": 9},
            {"id": "TRACE_12", "text": "Killing lock PID directly caused transaction rollback; safe fix is ordering queries alphabetically in application code.", "type": "Resolution Pattern", "importance": 8},
            {"id": "RAW_LOG_11", "text": "PostgreSQL log dump: 2024-03-12 14:02:11 UTC [pid 9481]: ERROR: deadlock detected ... 80 lines of lock trees.", "type": "Ephemeral Log", "importance": 3}
        ],
        "new_reflection": "Rule: To permanently prevent 40P01 deadlocks, enforce strict alphabetical table acquisition order across all application transactions and set deadlock_timeout to 1000ms."
    }
}

selected_mission_key = st.selectbox("Select Autonomous Agent Mission:", list(agent_scenarios.keys()))
active_mission = agent_scenarios[selected_mission_key]

st.info(f"**Agent Objective**: `{active_mission['goal']}`")

col_ag_a, col_ag_b = col_ag_a, col_ag_b = st.columns([1, 1])

with col_ag_a:
    st.subheader("Stage 1 & 2: Pre-Planning Cognitive Recall")
    
    st.markdown("### Recalled Past Experiences (Pre-Action Vector Query):")
    for mem in active_mission["past_memories"]:
        if mem["importance"] >= 7:
            st.success(f"**[{mem['id']}] {mem['type']} (Importance: {mem['importance']}/10)**  \n*{mem['text']}*")
        else:
            st.caption(f"• **[{mem['id']}] {mem['type']}**: {mem['text']}")
            
    st.markdown("""
    *Agent Planner Impact*: The agent incorporates these recalled failure fixes into its planning prompt **before calling any tools**, 
    guaranteeing it avoids the ECR auth error and memory limits on its very first try!
    """)

with col_ag_b:
    st.subheader("Stage 3 & 4: Reflection & Strategic Forgetting")
    
    st.markdown("### Stage 3: Synthesized Consolidated Reflection:")
    st.info(f"💡 **Learned Invariant Rule**:  \n**{active_mission['new_reflection']}**")
    
    st.markdown("### Stage 4: Strategic Garbage Collection (Pruning):")
    st.dataframe(pd.DataFrame([
        {"Item ID": "RAW_LOG_01 / 11", "Action": "DELETED / PURGED", "Rationale": "Ephemeral terminal stdout (low importance = 2)"},
        {"Item ID": "TRACE_01 / 11", "Action": "ARCHIVED", "Rationale": "Consolidated into golden invariant rule"},
        {"Item ID": "CONSOLIDATED_RULE", "Action": "SAVED PERMANENTLY", "Rationale": "High importance (10/10) persistent behavioral rule"}
    ]), use_container_width=True, hide_index=True)
    
    st.success("Memory footprint optimized! Ephemeral bloat removed; high-value cognitive rules preserved.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: How does an autonomous agent use vector embeddings for 'Pre-Planning Cognitive Recall'?", expanded=True):
    st.markdown("""
    **Answer:**
    In autonomous agent loops (e.g., ReAct, Plan-and-Solve, LangGraph), the agent forms an execution plan before executing tools.
    
    **The Cognitive Recall Stage**:
    1. Before generating the plan, the agent embeds its current task description: $v_{task} = \\text{Model}(\\text{goal})$.
    2. It executes an ANN cosine similarity query against an **Episodic Vector Memory Store** containing past trial-and-error logs:
       `past_trials = memory_store.search(v_task, top_k=3)`
    3. The retrieved episodic records (containing past tool errors, invalid API parameters, and successful workarounds) are injected directly into the agent's system prompt:
       * *Prompt*: `"You previously attempted a similar task. Note these past errors and avoid repeating them: {past_trials}"`
    4. This prevents the agent from repeating expensive, failed actions, dramatically reducing API trial loops and execution latency.
    """)

with st.expander("Q2: What is the Generative Reflection Tree in autonomous agent memory?"):
    st.markdown("""
    **Answer:**
    The **Generative Reflection Tree (Park et al., 2023 / MemGPT)** is a hierarchical memory abstraction mechanism:
    * **Level 1 (Leaves - Raw Observations)**: Low-level sensory inputs and raw tool outputs (e.g., bash stdout, curl responses, SQL error logs).
    * **Level 2 (Internal Nodes - Reflection Insights)**: An LLM analyzes batches of Level 1 observations to synthesize intermediate questions and insights (e.g., *"Why did the deployment fail? Because the port was already bound."*).
    * **Level 3 (Root - High-Level Invariant Rules)**: Top-level consolidated rules (e.g., *"Always check port availability with netstat before launching the server."*).
    * The agent queries high-level reflections first during planning, only drilling down into raw logs when fine-grained diagnostic debugging is required.
    """)

with st.expander("Q3: Why is 'Strategic Forgetting' (Garbage Collection) mandatory for long-running autonomous agents?"):
    st.markdown("""
    **Answer:**
    Without strategic forgetting, autonomous agents experience **Memory Poisoning & Retrieval Saturation**:
    1. **Context Bloat & Token Limits**: Agents generate thousands of intermediate tool tokens per hour. Storing everything saturates vector databases.
    2. **Retrieval Noise (Low SNR)**: When an agent queries its memory, search results get clogged with 50 pages of irrelevant curl logs from yesterday rather than the high-level solution.
    3. **Economic Cost**: Vector database storage and embedding API costs scale linearly with vector volume.
    * **Strategic Forgetting Policies** (TTL expiration on raw tool stdout, LRU cache eviction, and consolidating 20 raw logs into 1 high-level reflection while deleting the raw children) maintain high Signal-to-Noise Ratio (SNR) indefinitely.
    """)

with st.expander("Q4: How does LangGraph integrate with Vector Memory for state persistence?"):
    st.markdown("""
    **Answer:**
    In **LangGraph**, agent workflows are modeled as a state graph where nodes represent agent reasoning steps or tool calls:
    * **Short-Term State (Checkpointers)**: LangGraph uses checkpointers (`MemorySaver`, `SqliteSaver`, `PostgresSaver`) to serialize the current graph execution state (`messages`, `current_step`, `scratchpad`) across turns.
    * **Long-Term Memory (External Vector Store)**: When a graph node completes an execution trace, it calls an external vector store (Chroma/Qdrant) to persist episodic reflections keyed by `user_id` or `workspace_id`.
    * When a new thread starts, the graph's initialization node queries the vector store to hydrate the initial state with historical rules and user preferences.
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Storing Full Raw Tool Payloads Unconditionally**: Storing a 50,000-character SQL table dump in the vector memory. The vector becomes a fuzzy, uninformative blur of thousands of rows. Always summarize tool outputs before vectorization.
* **Trap 2: Indefinite Memory Retention Without TTL**: Keeping ephemeral API tokens, temporary passwords, and container IDs in agent memory indefinitely. The agent will attempt to reuse expired session tokens 3 months later. Always enforce TTL policies.
* **Trap 3: Forgetting Contradiction Resolution When Environment Rules Change**: If an API endpoint is updated from `/v1/auth` to `/v2/auth`, an agent with an un-updated reflection memory will stubbornly call the deprecated `/v1/auth` forever. Ensure memories have update and deprecation mechanisms.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: Cognitive Recall"):
        st.markdown("**Front**: What is Cognitive Recall in autonomous agents?")
        st.divider()
        st.markdown("**Back**: Querying episodic vector memory before planning to retrieve past errors, tool fixes, and successful paths.")

with card_cols[1]:
    with st.popover("Card 2: Reflection Tree"):
        st.markdown("**Front**: What does the Reflection Tree do?")
        st.divider()
        st.markdown("**Back**: Synthesizes high-level invariant behavioral rules from dozens of low-level tool observation logs.")

with card_cols[2]:
    with st.popover("Card 3: Strategic Forgetting"):
        st.markdown("**Front**: Why must autonomous agents strategically forget?")
        st.divider()
        st.markdown("**Back**: Purging ephemeral tool logs prevents vector memory bloat and protects search signal-to-noise ratio.")

with card_cols[3]:
    with st.popover("Card 4: LangGraph Memory"):
        st.markdown("**Front**: How does LangGraph separate short-term and long-term memory?")
        st.divider()
        st.markdown("**Back**: Checkpointers manage active graph thread execution; external vector stores maintain persistent cross-session reflections.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Autonomous Agent Memory:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the 4-stage agent memory lifecycle (Store, Recall, Reflect, Forget).", key="chk32_3_1")
    st.checkbox("I know how Cognitive Recall prevents agents from repeating past tool failures.", key="chk32_3_2")
    st.checkbox("I can implement an episodic memory manager with ChromaDB and OpenAI.", key="chk32_3_3")
with c2:
    st.checkbox("I understand the Reflection Tree and how to consolidate raw logs into rules.", key="chk32_3_4")
    st.checkbox("I know how Strategic Forgetting (TTL, LRU) prevents memory poisoning.", key="chk32_3_5")
    st.checkbox("I understand LangGraph checkpointer state vs. long-term vector memory stores.", key="chk32_3_6")
