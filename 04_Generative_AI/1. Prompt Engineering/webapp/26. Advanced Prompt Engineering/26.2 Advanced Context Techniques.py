import streamlit as st
import json

st.set_page_config(
    page_title="26.2 Advanced Context Techniques - Prompt Engineering",
    page_icon="🗜️",
    layout="wide"
)

# Header Section
st.title("26.2 Advanced Context Engineering & Long-Window Techniques")
st.caption("Chapter 26: Advanced Prompt Engineering | Domain: 1. Prompt Engineering")
st.markdown("""
Modern frontier models support 128,000 to over 2,000,000 tokens of context. 
However, simply dumping large unpruned documents into the prompt window triggers **Lost-in-the-Middle attention decay, 
hallucinations, quadratic latency spikes, and ballooning token bills**. Advanced context engineering treats context as a 
**dynamic, compressed, prioritized memory buffer**.
""")

st.divider()

# Section 1: The Lost-in-the-Middle Phenomenon
st.header("1. The 'Lost-in-the-Middle' Attention Topology", divider="rainbow")
st.markdown("""
Transformer attention heads do not distribute focus uniformly across long contexts. 
Extensive empirical research reveals a U-shaped retrieval accuracy curve:
* **Primacy Effect**: Information placed at the very beginning of the context (system instructions, first 10% of tokens) is recalled with high accuracy.
* **Recency Effect**: Information placed at the very end of the context (user query, final few-shot examples) enjoys near-perfect recall.
* **The Valley of Death (Middle 50-80%)**: Information buried in the middle of a 100k-token prompt suffers up to a 40% drop in retrieval accuracy.
""")

st.graphviz_chart('''
digraph ContextTopology {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10];
    
    Start [label="Beginning of Prompt\\n(High Attention / Primacy)\\nSystem Persona & Core Rules", fillcolor="#C8E6C9"];
    Middle [label="Middle of Context\\n(VALLEY OF DEATH - LOW ATTENTION)\\nAvoid placing critical needles here!", fillcolor="#FFCDD2"];
    End [label="End of Prompt\\n(High Attention / Recency)\\nUser Query & Most Relevant Docs", fillcolor="#C8E6C9"];
    
    Start -> Middle -> End;
}
''', use_container_width=False)

st.divider()

# Section 2: Six Advanced Context Techniques
st.header("2. Six Core Context Engineering Strategies", divider="rainbow")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Context Compression (LLMLingua)")
    st.markdown("""
    * **Concept**: Uses a small, high-speed language model (e.g., Llama-3-8B or GPT-2) to compute token perplexity across the prompt. 
    Removes low-information, redundant tokens while retaining semantic keywords.
    * **Impact**: Compresses prompts by 3x to 5x with minimal loss in reasoning accuracy, saving 70% on token costs.
    """)

    st.subheader("2. Strategic Context Prioritization")
    st.markdown("""
    * **Concept**: Never inject retrieved chunks in random order. Sort chunks by relevance score and place the **highest-scoring chunks at the very beginning and very end** of the context block.
    """)

    st.subheader("3. Dynamic Context Selection")
    st.markdown("""
    * **Concept**: Adjust the retrieved token budget dynamically based on query complexity. 
    Simple factoid questions receive 1-2 chunks; complex comparative queries receive 10 chunks.
    """)

with col2:
    st.subheader("4. Semantic Context Selection (Thresholding)")
    st.markdown("""
    * **Concept**: Enforce hard cosine similarity cutoffs (e.g., `similarity >= 0.78`). 
    If only 1 chunk meets the threshold, inject only 1 chunk rather than forcing arbitrary top-k padding.
    """)

    st.subheader("5. Memory-Aware Context Slicing")
    st.markdown("""
    * **Concept**: Maintain separate buffers for **Episodic Memory** (past events), **Working Memory** (current active task), and **Entity Memory** (user profile). 
    Assemble context dynamically per turn.
    """)

    st.subheader("6. Long-Context Needle Framing")
    st.markdown("""
    * **Concept**: In massive 500k+ token prompts, anchor critical reference data with explicit XML delimiters and clear framing instructions:
    `<critical_facts priority="highest"> ... </critical_facts>`.
    """)

st.divider()

# Section 3: Strategic Chunk Re-ordering Implementation
st.header("3. Implementation: Optimal U-Shape Chunk Placement", divider="rainbow")
st.markdown("""
The Python helper below reorganizes ranked RAG chunks into an optimal U-shape to conquer the Lost-in-the-Middle phenomenon:
""")

st.code('''# U-Shaped Context Placement Algorithm
def order_chunks_for_optimal_attention(ranked_chunks: list[str]) -> list[str]:
    """
    Distributes highest-ranked chunks to the start and end of the context
    to counteract the Lost-in-the-Middle attention valley.
    """
    if len(ranked_chunks) <= 2:
        return ranked_chunks
        
    start_chunks = []
    end_chunks = []
    
    # Alternating placement: Rank 1 -> End, Rank 2 -> Start, Rank 3 -> End, etc.
    for i, chunk in enumerate(ranked_chunks):
        if i % 2 == 0:
            end_chunks.insert(0, chunk)  # Best chunks closer to the user query
        else:
            start_chunks.append(chunk)   # Second best at the very top
            
    # Assembles: [Rank 2, Rank 4, ..., Middle Chunks, ..., Rank 3, Rank 1]
    return start_chunks + end_chunks''', language="python")

st.divider()

# Section 4: Interactive Context Sizer & Simulator
st.header("4. Interactive Context Optimization Workbench", divider="rainbow")
st.markdown("Experiment with how context compression and strategic placement protect generation quality.")

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    raw_tokens = st.slider("Raw Ingested Document Size (Tokens)", 5000, 100000, 40000, step=5000)
    apply_compression = st.checkbox("Enable Perplexity-Based Compression (LLMLingua)", value=True)
    apply_ushape = st.checkbox("Apply U-Shaped Attention Placement", value=True)

with col_sim2:
    compression_ratio = 0.35 if apply_compression else 1.0
    effective_tokens = int(raw_tokens * compression_ratio)
    
    # Standard GPT-4o input cost: $2.50 per 1M tokens
    cost_per_req = (effective_tokens / 1_000_000) * 2.50
    raw_cost = (raw_tokens / 1_000_000) * 2.50
    savings_pct = (1.0 - (effective_tokens / raw_tokens)) * 100
    
    st.subheader("Context Telemetry Analysis")
    m1, m2 = st.columns(2)
    m1.metric("Effective Prompt Tokens", f"{effective_tokens:,}", delta=f"-{savings_pct:.0f}%" if apply_compression else "0%")
    m2.metric("Cost per Request", f"${cost_per_req:.4f}", delta=f"-${raw_cost - cost_per_req:.4f}" if apply_compression else "$0.00")
    
    if apply_ushape:
        st.success("✅ U-Shape Optimization Active: Critical chunks pinned to primacy and recency boundaries.")
    else:
        st.warning("⚠️ Linear Placement: Risk of needle-in-haystack degradation in middle tokens.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa1, qa2 = st.columns(2)

with qa1:
    with st.expander("Q: What causes the 'Lost-in-the-Middle' phenomenon in Transformer architectures?"):
        st.write("""
        Transformer self-attention computes pairwise token interactions across the entire sequence. 
        However, training objectives (causal language modeling) and positional encoding schemes (such as RoPE or ALiBi) 
        naturally create attention bias toward recent tokens (recency) and initial prompt instructions (primacy). 
        Consequently, tokens positioned between 40% and 80% of the sequence receive lower aggregate attention weights, 
        making factual retrieval from the middle significantly more prone to hallucination.
        """)

    with st.expander("Q: How does LLMLingua achieve prompt compression without losing reasoning ability?"):
        st.write("""
        LLMLingua uses a smaller, faster model (e.g. Llama-3-8B) to compute token perplexity across the candidate prompt. 
        Syntactic filler words, redundant formatting, and low-entropy tokens have low conditional surprise and can be 
        safely pruned. Highly informative keywords, entity names, and distinct numbers carry high perplexity and are preserved. 
        This selective pruning preserves semantic density while shedding 50% to 70% of total tokens.
        """)

with qa2:
    with st.expander("Trap to Avoid: Blind Top-K Padding Without Similarity Thresholds"):
        st.write("""
        Setting a fixed `k=10` retrieval window means that when a user asks an obscure question with only 1 relevant doc in the database, 
        the vector search will inject 9 completely irrelevant chunks to fill the quota. 
        These 9 irrelevant chunks pollute the context and cause hallucinations. 
        Always enforce a minimum semantic similarity threshold (`score >= 0.75`) to drop low-quality distractors.
        """)

    with st.expander("Flashcards: Context Techniques"):
        fc_sel = st.selectbox(
            "Select Technique Flashcard:",
            ["U-Shaped Placement", "Perplexity Compression", "Semantic Thresholding", "Working Memory Buffer"]
        )
        if fc_sel == "U-Shaped Placement":
            st.info("Ordering context chunks so the most relevant items reside at the start and end of the prompt.")
        elif fc_sel == "Perplexity Compression":
            st.info("Pruning low-entropy tokens using smaller models to slash token size without losing meaning.")
        elif fc_sel == "Semantic Thresholding":
            st.info("Dropping retrieved chunks that fall below a strict cosine similarity score threshold.")
        else:
            st.info("A dynamic context slice containing only currently active goals and constraints.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the causes and solutions for the Lost-in-the-Middle attention phenomenon.")
    st.checkbox("I can implement a U-shaped chunk ordering function in Python.")
with c2:
    st.checkbox("I know how perplexity-based compression (LLMLingua) reduces prompt token budgets.")
    st.checkbox("I understand why fixed top-k padding without similarity thresholds harms generation.")
