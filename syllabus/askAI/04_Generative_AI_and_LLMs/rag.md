Yes. If you're going to give the AI **only one RAG section at a time**, use this as your reusable **master Retrieval-Augmented Generation prompt**. Each time, just replace `[RAG SECTION]`.

I am learning Retrieval-Augmented Generation (RAG) as part of my AI/ML and Generative AI learning journey.

I will give you **one RAG roadmap section at a time**. Your job is to teach me that section thoroughly, from fundamentals to advanced understanding.

Assume that I know basic Python, have some exposure to LLMs, embeddings, and vector databases, but I want to develop a **strong practical and conceptual understanding** of RAG — how it works, why it works, and how to build production-grade RAG systems.

## RAG Section

[RAG SECTION]

---

# How I want you to teach me

For every topic and subtopic in the section, follow this learning structure:

### 1. What is it?

- Give a clear definition.
- Explain the problem it solves (hallucination, knowledge cutoff, private data access).
- Explain why RAG exists as an alternative to fine-tuning or longer context windows.
- Place it in the RAG pipeline stage: Ingestion → Retrieval → Augmentation → Generation.

### 2. Intuition

- Explain the concept in very simple language.
- Use a real-world analogy (e.g., RAG is like an open-book exam — the LLM reads relevant notes before answering).
- Explain what problem this specific RAG component solves within the pipeline.
- Then gradually move toward the technical explanation.

### 3. Prerequisites

- Mention any Python/NLP/LLM/vector database concepts I need to understand first.
- If necessary, briefly teach the prerequisite before continuing.

### 4. Architecture and Core Concepts

- Draw a clear text diagram of this RAG component and where it fits in the full pipeline:

```
Documents → Chunking → Embedding → Vector Store
                                        ↓
Query → Query Embedding → Retrieval → Context Assembly → LLM → Answer
```

- Explain the role of every component involved.
- Show the important code (using LangChain, LlamaIndex, or raw Python as appropriate).
- Explain important parameters and configuration options.

### 5. Examples

Give examples progressively:

**Conceptual → Basic Implementation → Advanced / Production Pattern**

For every important example:

- Show the code.
- Explain it step by step.
- Show the expected output or behavior.
- Explain why the output is produced.
- Explain what would go wrong if this step were skipped or done incorrectly.

Use realistic use cases: Q&A over PDF documents, internal knowledge bases, customer support bots, code assistants.

### 6. Under the Hood

When relevant, explain what is happening internally at each RAG stage.

For example:

- How chunk boundaries affect retrieval quality
- How cosine similarity is computed between query and document embeddings
- How the context is assembled and injected into the prompt
- How the LLM uses (or ignores) the retrieved context
- How re-ranking models score and reorder retrieved chunks
- How HyDE (Hypothetical Document Embeddings) improves retrieval
- How multi-query retrieval generates multiple query variants

### 7. RAG Failure Modes and Debugging

Explain what can go wrong at each RAG stage and how to diagnose it:

| Stage      | Failure               | Symptom                              | Fix                             |
| ---------- | --------------------- | ------------------------------------ | ------------------------------- |
| Chunking   | Chunks too large      | Irrelevant context retrieved         | Smaller chunk size              |
| Embedding  | Wrong embedding model | Poor retrieval                       | Domain-specific embedder        |
| Retrieval  | Low top-k             | Missing relevant chunks              | Increase k or use hybrid search |
| Generation | Context ignored       | Hallucination despite good retrieval | Better prompt template          |

Go deep on the failure modes relevant to the section being taught.

### 8. Advanced RAG Techniques

When relevant to the section, cover advanced patterns:

- **Naive RAG → Advanced RAG → Modular RAG** evolution
- **Hybrid Search**: combining dense (vector) and sparse (BM25/TF-IDF) retrieval
- **Re-ranking**: using cross-encoder models (e.g., Cohere Rerank, BGE-reranker) after initial retrieval
- **HyDE**: generating a hypothetical answer to improve query embedding
- **Multi-query Retrieval**: generating multiple query variants
- **Self-RAG**: the model decides whether to retrieve and whether to use retrieved content
- **RAPTOR**: recursive abstractive processing for tree-organized retrieval
- **Agentic RAG**: using an LLM agent to iteratively retrieve and reason

Only cover techniques directly relevant to the section.

### 9. Evaluation

Explain how to evaluate this RAG component:

- **Retrieval metrics**: Recall@k, Precision@k, MRR, NDCG
- **Generation metrics**: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- **End-to-end metrics**: RAGAS framework, TruLens, DeepEval
- How to set up a golden dataset for RAG evaluation
- What "faithfulness" means and why it is not the same as accuracy

Show code examples using RAGAS or a similar evaluation framework where applicable.

### 10. Performance and Cost

Explain performance and cost considerations when relevant:

- Embedding cost (API calls per document vs. local embedding model)
- Vector store query latency at different scales
- Re-ranking latency vs. retrieval quality trade-off
- LLM token cost — how context length affects cost
- Caching strategies (embedding cache, retrieval cache, generation cache)
- Chunking strategy impact on total token usage

### 11. Common Mistakes

Show common mistakes beginners/intermediate learners make in RAG systems.

For each mistake:

**Wrong approach → Why it is wrong → Correct approach**

Also mention important pitfalls such as:

- Using the wrong chunk size for the embedding model's context window
- Not cleaning documents before embedding (e.g., noisy PDF text)
- Retrieving too few or too many chunks
- Stuffing too much context into the prompt
- Not filtering retrieved chunks by relevance score threshold
- Evaluating RAG only by end-user satisfaction and not by retrieval quality
- Using the same embedding model for indexing and a different one for querying

### 12. Comparison

Compare different RAG approaches or components in a table.

For example:

| Approach | How it works | Advantages | Disadvantages | When to use |
| -------- | ------------ | ---------- | ------------- | ----------- |

### 13. Best Practices

Give practical rules I should follow when building real production RAG systems.

Include:

- Always evaluate retrieval quality independently from generation quality
- Use metadata filtering to narrow retrieval before vector search
- Version your document index and embedding model together
- Test chunk size and overlap empirically on your specific document corpus
- Use hybrid search for production systems
- Any section-specific best practices

### 14. Interview Perspective

Give me important interview questions related to the section.

Include:

- Conceptual questions (e.g., "What is the difference between RAG and fine-tuning?")
- Architecture questions (e.g., "How would you handle documents that are longer than the context window?")
- Debugging questions (e.g., "Your RAG system retrieves correct chunks but still hallucinates — why?")
- Engineering questions (e.g., "How would you scale a RAG system to 10 million documents?")
- Evaluation questions (e.g., "How do you measure faithfulness in a RAG system?")

Provide the answers and explanations after the questions.

### 15. Practice

Give me exercises at three levels:

**Beginner**

- 2–3 exercises

**Intermediate**

- 2–3 exercises

**Advanced**

- 2–3 exercises

Do NOT immediately provide solutions.

Let me solve them first.

When I submit my answers:

- Check them.
- Identify mistakes.
- Explain why they are wrong.
- Show the correct solution.
- Suggest a better approach if one exists.

### 16. Mini Project

Give me a small RAG mini-project using the concepts in this section.

Give me:

- Problem statement (e.g., build a Q&A bot over a set of PDF documents)
- Tech stack to use
- Requirements
- Expected behavior
- Constraints (e.g., must use open-source embedding model, must run locally)
- Hints

Do not provide the complete solution until I attempt it.

---

# Important Teaching Rules

1. Do not just give definitions. Teach the reasoning behind every RAG design decision.

2. Always connect individual RAG components to the full pipeline — nothing exists in isolation.

3. Start simple (naive RAG) and gradually increase complexity (advanced RAG, agentic RAG).

4. When showing retrieval, always show the actual query vector and retrieved chunk content.

5. When showing chunking, always show the actual before/after text with chunk boundaries.

6. Always address failure modes — knowing what can go wrong is as important as knowing how it works.

7. When performance is relevant, give concrete numbers (e.g., "embedding 1M chunks with text-embedding-3-small costs ~$0.02").

8. Do not introduce unrelated RAG topics.

9. If an advanced concept depends on another concept, explain the dependency first.

10. Do not assume that I understand vector databases, embedding models, or LLM prompting deeply.

11. Use tables, pipeline diagrams, and chunk/embedding visualizations when they make concepts easier to understand.

12. Distinguish clearly between:

- Naive RAG (basic retrieve-then-generate)
- Advanced RAG (query transformation, re-ranking, hybrid search)
- Modular RAG (composable pipeline components)
- Agentic RAG (LLM decides when and what to retrieve)

13. Relate every concept to real production RAG use cases (document Q&A, knowledge bases, customer support).

14. Do not teach the entire RAG roadmap at once.

15. Teach ONLY the section I provide.

16. At the end, give me a concise **Revision Sheet** containing:

- Key concepts
- Pipeline diagram summary
- Important parameters and configuration choices
- Failure modes and fixes
- Evaluation metrics summary
- Best practices checklist
- Common mistakes

17. After teaching the section, STOP and wait for me. Do not automatically move to another RAG section.

---

# Teaching Goal

My goal is not simply to call a RAG API.

I want to understand:

**What → Why → Architecture → Internals → Failure Modes → Evaluation → Production**

so that I can confidently design, implement, debug, evaluate, and scale RAG systems for real Generative AI applications.

Now teach me the following RAG section:

[RAG SECTION]

**How to use it:** paste the prompt once, then replace `[RAG SECTION]` with something like `3. Chunking Strategies` or a smaller subsection such as `3.1 Recursive Character Text Splitter vs. Semantic Chunking`. This keeps each lesson focused and prevents the AI from overwhelming you.
