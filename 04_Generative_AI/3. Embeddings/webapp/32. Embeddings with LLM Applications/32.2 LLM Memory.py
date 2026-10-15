import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import time
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="32.2 LLM Memory - Embeddings Curriculum",
    page_icon="🧠",
    layout="wide"
)

# Title & Overview
st.title("32.2 LLM Memory: Conversation Embeddings, Long-Term Semantic Memory & Contextual Recall")
st.caption("Chapter 32: Embeddings with LLM Applications | Domain: 3. Vector Embeddings")
st.markdown("""
By default, Large Language Models are **completely stateless**: 
each API request starts with a blank slate, with zero recollection of previous conversations, user preferences, or past interactions.

While short-term memory can be maintained by appending recent conversation turns to the chat history, 
this approach hits hard limits:
* **Token Window Limits**: Sliding chat buffers quickly exhaust context windows and balloon API token costs.
* **Session Boundaries**: Closing a browser tab or starting a new session resets the chat history to zero.
* **Lack of Associative Retrieval**: A user's dietary restriction mentioned 3 weeks ago cannot be retrieved by a naive sliding window.

**Embedding-Based Long-Term Memory** solves this by treating every conversation turn and extracted user preference 
as an embedded memory record stored in an external vector store. 
When a user asks a question, the system retrieves only the **semantically relevant memories from weeks or months ago**, 
enabling deeply personalized, lifelong conversational AI assistants.
""")

st.divider()

# Section 1: The Three Types of LLM Memory
st.header("1. Cognitive Architecture: The 3 Tiers of LLM Memory", divider="rainbow")

col_tier1, col_tier2 = col_tier1, col_tier2 = st.columns([1, 1])

with col_tier1:
    st.markdown("""
    ### The 3 Tiers of Cognitive Memory
    Inspired by human cognitive neuroscience, modern conversational architectures implement three memory tiers:
    
    1. **Working Memory (In-Context Buffer)**:
       * The last 2 to 5 chat turns passed directly inside the LLM prompt.
       * Immediate conversational continuity (*"What did you say right before this?"*).
    2. **Episodic Memory (Timestamped Event Stream)**:
       * Vectorized interaction logs recording specific past events, user questions, and assistant responses tagged with timestamps:
         * `{ "timestamp": "2024-03-12", "event": "User planned trip to Tokyo with budget $3000" }`
    3. **Semantic Memory (Structured Fact Store)**:
       * Generalized personal facts, preferences, and entity knowledge extracted by an LLM:
         * *Fact 1*: *"User is lactose intolerant and avoids dairy."*
         * *Fact 2*: *"User's primary programming language is Rust."*
         * *Fact 3*: *"User is based in Seattle, Washington (PST)."*
    """)

with col_tier2:
    st.markdown("### The Long-Term Memory Pipeline")
    mem_dot = """
    digraph LLMMemoryFlow {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        ChatTurn [label="User Chat Message\\n'Suggest a coffee shop near my office'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        subgraph cluster_memory_store {
            label="Long-Term Vector Memory Store (Chroma / Qdrant)"; style="dashed"; color="#10B981";
            Mem1 [label="Memory: 'User drinks oat milk lattes'\\n(Importance: 8/10, Age: 4 days)", fillcolor="#D1FAE5"];
            Mem2 [label="Memory: 'User office is in Downtown Seattle'\\n(Importance: 9/10, Age: 2 weeks)", fillcolor="#D1FAE5"];
            Mem3 [label="Memory: 'User drives an electric car'\\n(Irrelevant to coffee prompt)", fillcolor="#F3F4F6"];
        }

        Retriever [label="Composite Memory Scorer\\nRelevance + Recency + Importance\\nFilters out Mem3", fillcolor="#FEF3C7", color="#F59E0B", style="filled,bold"];

        ContextPrompt [label="Personalized LLM Prompt\\nInjected Memories: [Mem1, Mem2]\\n+ Current Query", fillcolor="#FDE68A"];

        PersonalizedOutput [label="Personalized Output\\n'Here are 3 great cafes in Downtown Seattle\\nwith excellent oat milk options!'", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        ChatTurn -> Retriever [label="Query Vector"];
        Mem1 -> Retriever;
        Mem2 -> Retriever;
        Mem3 -> Retriever;
        Retriever -> ContextPrompt;
        ContextPrompt -> PersonalizedOutput;
    }
    """
    st.graphviz_chart(mem_dot, use_container_width=False)

st.divider()

# Section 2: Composite Memory Scoring: Relevance, Recency & Importance
st.header("2. The Generative Agents Memory Scoring Formula", divider="rainbow")

st.markdown("""
Pioneered by Stanford researchers in the landmark paper **Generative Agents (Park et al., 2023)**, 
retrieving long-term memory requires balancing three distinct dimensions rather than relying purely on cosine similarity:

```text
Memory_Score = (w_rel * Relevance) + (w_rec * Recency) + (w_imp * Importance)
```

1. **Semantic Relevance (Cosine Similarity)**:
   * How close is the memory vector to the current prompt vector? (`Relevance in [0.0, 1.0]`).
2. **Recency (Exponential Time Decay)**:
   * Memories decay exponentially over time unless reinforced:
     `Recency = exp( -lambda * delta_t_hours )`
   * Prevents memories from 2 years ago from overpowering fresh context.
3. **Importance Score (LLM-Evaluated Saliency)**:
   * Not all memories are equal. An LLM rates memory significance on a scale of 1 to 10:
     * *"User prefers oat milk"* -> **Importance: 8** (high personal preference).
     * *"User said the weather is nice today"* -> **Importance: 2** (ephemeral trivia).
""")

col_sc1, col_sc2 = col_sc1, col_sc2 = st.columns([1, 1])

with col_sc1:
    st.markdown("### Production Code: Memory Extraction Engine")
    st.code("""
from openai import OpenAI

client = OpenAI()

EXTRACT_FACTS_PROMPT = \"\"\"Analyze the following conversation turn and extract any 
long-term facts, personal preferences, dietary needs, or project details about the user.
Assign each fact an importance rating from 1 (ephemeral) to 10 (critical personal trait).

Format as JSON list:
[{"fact": "...", "importance": 8}]

Conversation Turn:
User: {user_msg}
Assistant: {assistant_msg}\"\"\"

def extract_and_store_memory(user_msg: str, assistant_msg: str, memory_collection):
    # 1. Extract discrete facts
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": EXTRACT_FACTS_PROMPT.format(user_msg=user_msg, assistant_msg=assistant_msg)}],
        response_format={"type": "json_object"}
    )
    # 2. Embed each fact and store in ChromaDB with timestamp and importance metadata
""", language="python")

with col_sc2:
    st.markdown("### Production Code: Composite Retrieval Formula")
    st.code("""
import numpy as np
import time

def compute_memory_score(
    cosine_sim: float, 
    created_timestamp: float, 
    importance_rating: int,
    half_life_hours: float = 72.0,
    w_rel: float = 0.5,
    w_rec: float = 0.25,
    w_imp: float = 0.25
) -> float:
    # 1. Recency decay
    delta_hours = (time.time() - created_timestamp) / 3600.0
    recency = 0.5 ** (delta_hours / half_life_hours)
    
    # 2. Normalized importance (1-10 -> 0.1-1.0)
    norm_importance = importance_rating / 10.0
    
    # 3. Composite linear score
    return (w_rel * cosine_sim) + (w_rec * recency) + (w_imp * norm_importance)
""", language="python")

st.divider()

# Section 3: Interactive Workbench — Lifelong Memory & Recall Simulator
st.header("3. Interactive Workbench: Lifelong Memory Retrieval Simulator", divider="rainbow")

st.markdown("""
Explore how an LLM recalls personal user memories across multiple past sessions. 
Below is a user's persistent **Long-Term Memory Bank** collected across the last 30 days. 
Type a prompt to watch the retrieval engine score memories by **Relevance + Recency + Importance** and synthesize a personalized response!
""")

simulated_memory_bank = [
    {
        "id": "MEM_01",
        "fact": "User is strictly lactose intolerant and prefers oat milk or almond milk substitutes.",
        "category": "Dietary",
        "days_ago": 3,
        "importance": 9,
        "keywords": ["milk", "lactose", "diet", "coffee", "dairy", "breakfast", "food"]
    },
    {
        "id": "MEM_02",
        "fact": "User's physical office is located in Downtown Seattle near Pioneer Square.",
        "category": "Location",
        "days_ago": 12,
        "importance": 8,
        "keywords": ["office", "seattle", "location", "work", "downtown", "commute", "neighborhood"]
    },
    {
        "id": "MEM_03",
        "fact": "User is a Senior Staff Engineer building distributed microservices in Rust and Go.",
        "category": "Career",
        "days_ago": 20,
        "importance": 7,
        "keywords": ["rust", "golang", "engineer", "software", "microservices", "code", "programming"]
    },
    {
        "id": "MEM_04",
        "fact": "User has a 3-year-old Golden Retriever named Cooper who loves dog parks.",
        "category": "Personal",
        "days_ago": 5,
        "importance": 6,
        "keywords": ["dog", "pet", "golden retriever", "park", "cooper", "animal"]
    },
    {
        "id": "MEM_05",
        "fact": "User mentioned having a mild headache during a chat session last Tuesday.",
        "category": "Ephemeral",
        "days_ago": 14,
        "importance": 2,
        "keywords": ["headache", "tired", "health", "tuesday"]
    }
]

prompt_presets = [
    "Can you recommend a good cafe or breakfast spot to meet a coworker near my office?",
    "Suggest a good outdoor weekend activity I can do with my pet.",
    "Recommend an advanced technical book or project idea for my weekend coding.",
    "What kind of quick healthy snacks should I stock in my pantry?"
]

selected_prompt = st.selectbox("Choose a User Prompt or Enter Custom:", prompt_presets + ["Custom Prompt"])
if selected_prompt == "Custom Prompt":
    active_user_prompt = st.text_input("Enter prompt:", value="Where should I grab coffee tomorrow morning?")
else:
    active_user_prompt = selected_prompt

col_mem_calc, col_mem_resp = col_mem_calc, col_mem_resp = st.columns([1, 1])

# Memory scoring function
def score_all_memories(prompt: str, memory_bank: list[dict]):
    p_lower = prompt.lower()
    scored = []
    
    for mem in memory_bank:
        # Calculate keyword semantic match
        overlap = sum(1 for kw in mem["keywords"] if kw in p_lower)
        relevance = min(0.96, round(0.20 + (overlap * 0.25), 3))
        if overlap == 0:
            relevance = 0.15
            
        # Recency decay (half-life: 7 days)
        recency = round(0.5 ** (mem["days_ago"] / 7.0), 3)
        
        # Importance
        imp_norm = mem["importance"] / 10.0
        
        # Composite score: 50% relevance, 25% recency, 25% importance
        total_score = round((0.50 * relevance) + (0.25 * recency) + (0.25 * imp_norm), 3)
        
        scored.append({
            **mem,
            "relevance": relevance,
            "recency": recency,
            "total_score": total_score
        })
    return sorted(scored, key=lambda x: x["total_score"], reverse=True)

scored_memories = score_all_memories(active_user_prompt, simulated_memory_bank)

with col_mem_calc:
    st.subheader("Composite Memory Ranking")
    
    st.markdown("### Top Retrieved Long-Term Memories:")
    df_mem_disp = pd.DataFrame([
        {
            "Memory ID": m["id"],
            "Fact": m["fact"][:55] + "...",
            "Rel (50%)": f"{m['relevance']:.2f}",
            "Rec (25%)": f"{m['recency']:.2f}",
            "Imp (25%)": f"{m['importance']}/10",
            "Composite": f"{m['total_score']:.3f}"
        }
        for m in scored_memories
    ])
    st.dataframe(df_mem_disp, use_container_width=True, hide_index=True)
    
    st.caption("""
    *Scoring Breakdown*: Memories with high semantic overlap + high importance + recent timestamps dominate the top ranks!
    Notice how ephemeral trivia (Headache, MEM_05) is automatically deprioritized.
    """)

# Synthesize response
top_2_memories = scored_memories[:2]

with col_mem_resp:
    st.subheader("Personalized LLM Generation")
    
    st.markdown("**Injected Context Memory Payload:**")
    for tm in top_2_memories:
        st.info(f"**[{tm['id']}]** *{tm['fact']}* (Score: {tm['total_score']:.3f})")
        
    st.markdown("### Synthesized Response:")
    if "cafe" in active_user_prompt.lower() or "coffee" in active_user_prompt.lower():
        st.success("""
        Here are two excellent cafes near **Downtown Seattle (Pioneer Square)** that are great for a work catch-up and feature fantastic **oat and almond milk alternatives**:

        1. **Elm Coffee Roasters (Pioneer Square)**: Just a short walk from your office, featuring light-roast single-origin espressos and house-made oat milk in a bright, spacious setting.
        2. **Zeitgeist Coffee**: A classic Pioneer Square workspace favorite with ample seating, great drip coffee, and dairy-free pastry options.
        """)
    elif "pet" in active_user_prompt.lower() or "outdoor" in active_user_prompt.lower():
        st.success("""
        For a great outdoor weekend outing with **Cooper**, I recommend:
        * **Magnuson Park Off-Leash Area**: The premier dog park in Seattle with direct freshwater beach access where Golden Retrievers love swimming and playing fetch!
        * **Marymoor Park (Redmond)**: A 40-acre dog playground with wide open fields and river access.
        """)
    elif "code" in active_user_prompt.lower() or "rust" in active_user_prompt.lower() or "book" in active_user_prompt.lower():
        st.success("""
        Given your focus on high-performance distributed systems in **Rust and Go**, here are two standout weekend project ideas:
        * **Build a Lightweight Raft Consensus Engine in Rust**: Implement leader election, log replication, and persistence using Tokio and Serde.
        * **Custom Distributed In-Memory Vector Index**: Build an HNSW graph index in Go with memory-mapped files and SIMD dot products.
        """)
    else:
        st.success(f"""
        Based on your preference for **dairy-free options** and your active schedule:
        * Great options include overnight oats with chia seeds and almond milk, or high-protein vegan trail mix!
        """)

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: Why is an external vector store required for LLM memory instead of relying on a 1-million token context window?", expanded=True):
    st.markdown("""
    **Answer:**
    While modern models (e.g., Gemini 1.5 Pro) support 1M+ token context windows, stuffing an entire year of conversation history into the prompt fails on four dimensions:
    1. **Economic Cost**: Running a 500,000-token prompt on every user message costs several dollars per chat turn. Over a month, a single active user would incur thousands of dollars in inference bills!
    2. **Inference Latency**: Processing 500k input tokens takes several seconds to generate the first token (Time To First Token - TTFT), ruining real-time conversational chat responsiveness.
    3. **Attention Degradation (Lost in the Middle)**: With hundreds of thousands of tokens, self-attention gets noisy, increasing the risk of hallucinations or ignoring subtle personal preferences.
    4. **Privacy & Security Boundaries**: An external vector database allows granular memory deletion (GDPR Right to Be Forgotten) and tenant isolation, which is impossible inside a monolithic raw prompt history.
    """)

with st.expander("Q2: How does the Exponential Recency Decay function work in memory retrieval?"):
    st.markdown("""
    **Answer:**
    The exponential recency decay function models the natural forgetting curve:
    ```text
    Recency = 0.5 ** (delta_t_hours / half_life_hours)
    ```
    * `delta_t_hours` is the elapsed time since the memory was formed.
    * `half_life_hours` (e.g., 72 hours or 7 days) defines the duration after which the memory's recency score drops by 50%.
    * **Why it matters**: A temporary event (e.g., *"I have a flight tonight"*) should decay rapidly so the assistant doesn't ask about the flight next month. Conversely, persistent traits (e.g., dietary restrictions) are protected by high **Importance** ratings, keeping them retrievable even as recency decays.
    """)

with st.expander("Q3: How do you handle contradictory or outdated memories (e.g., user moves from Seattle to New York)?"):
    st.markdown("""
    **Answer:**
    Handling memory updates requires **Memory Reconciliation & Consolidation**:
    1. **Semantic Collision Check**: When a new fact is extracted (*"User recently relocated to New York City"*), the pipeline queries the memory vector store for existing memories with high cosine similarity under the same entity/category (`Location`).
    2. **Contradiction Resolution Prompt**: If an existing memory is found (*"User lives in Seattle"*), pass both statements to a fast reconciliation LLM:
       `"Does Fact B contradict or supersede Fact A? If superseded, output UPDATE directive."`
    3. **Soft-Delete / Archive**: Mark the older Seattle memory as `superseded_at = now()` and insert the new New York memory as active.
    4. This prevents the LLM from hallucinating that the user lives in two cities simultaneously.
    """)

with st.expander("Q4: How do you comply with GDPR 'Right to be Forgotten' in LLM memory systems?"):
    st.markdown("""
    **Answer:**
    Complying with GDPR Article 17 (Right to Erasure) is straightforward in vector memory systems:
    * Because memories are stored in an external vector database (Qdrant, Chroma, DynamoDB) with structured metadata `{ "user_id": "usr_123", "memory_id": "mem_456" }`, deleting all user data requires a single database call:
      `collection.delete(where={"user_id": "usr_123"})`
    * Conversely, if user knowledge had been baked directly into model weights via fine-tuning, machine unlearning would require completely deleting and retraining the entire neural network!
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Storing Full Raw Dialogues Instead of Extracted Facts**: Vectorizing entire raw chat turns (*"Hey bot, how are you? - I am good, how are you? - What is the weather?"*). The vector gets diluted by conversational fluff. Always extract structured atomic facts before embedding.
* **Trap 2: Omitting Topic Half-Life Customization**: Applying the same 3-day recency decay to dietary preferences as to temporary travel plans. The bot will forget that the user is allergic to peanuts after a week!
* **Trap 3: Memory Bloat Without Deduplication**: Inserting *"User likes coffee"* every morning for 6 months, creating 180 duplicate memory vectors that saturate the top-k retrieval window. Always deduplicate against existing memories before insertion.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: 3 Memory Tiers"):
        st.markdown("**Front**: What are the 3 tiers of LLM memory?")
        st.divider()
        st.markdown("**Back**: 1. Working Memory (in-context buffer), 2. Episodic Memory (event stream), 3. Semantic Memory (long-term facts and traits).")

with card_cols[1]:
    with st.popover("Card 2: Composite Score"):
        st.markdown("**Front**: What 3 dimensions score memory retrieval?")
        st.divider()
        st.markdown("**Back**: 1. Semantic Relevance (Cosine Sim), 2. Recency (Time Decay), 3. Importance (Saliency 1-10).")

with card_cols[2]:
    with st.popover("Card 3: Fact Extraction"):
        st.markdown("**Front**: Why extract discrete facts instead of embedding raw chats?")
        st.divider()
        st.markdown("**Back**: Raw chat has conversational noise; discrete facts are concise, dense, and provide precise semantic retrieval.")

with card_cols[3]:
    with st.popover("Card 4: Reconciliation"):
        st.markdown("**Front**: How are contradictory memories resolved?")
        st.divider()
        st.markdown("**Back**: Collision detection finds conflicting facts (e.g. moving cities) and archives the older superseded record.")

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master LLM Long-Term Memory:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the 3 tiers of cognitive memory (Working, Episodic, Semantic).", key="chk32_2_1")
    st.checkbox("I know the composite memory scoring formula (Relevance, Recency, Importance).", key="chk32_2_2")
    st.checkbox("I can implement an LLM-based fact extraction and embedding pipeline.", key="chk32_2_3")
with c2:
    st.checkbox("I understand why vector memory is required despite 1M+ token context windows.", key="chk32_2_4")
    st.checkbox("I know how to resolve contradictory and superseded memory records.", key="chk32_2_5")
    st.checkbox("I understand GDPR Right to be Forgotten compliance in vector stores.", key="chk32_2_6")
