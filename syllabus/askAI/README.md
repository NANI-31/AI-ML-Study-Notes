# 📚 AI/ML Study Prompts — Master Index

Each file in this folder is a **reusable master prompt** for studying one topic section at a time.
Paste the prompt into any AI assistant, replace the placeholder (e.g., `[NUMPY SECTION]`), and get a structured, deep lesson.

---

## How to Use

1. Open the `.md` file for the topic you are studying.
2. Copy the entire prompt.
3. Replace the `[TOPIC SECTION]` placeholder with the specific section you want to learn (e.g., `3.2 Broadcasting`).
4. Paste into your AI assistant and start learning.

---

## 📁 Folder Structure and Learning Path

```
askAI/
├── 01_Data_Science_Foundations/   ← Start here
├── 02_Machine_Learning/            ← Then here
├── 03_Data_Visualization/          ← In parallel with 01 & 02
├── 04_Generative_AI_and_LLMs/     ← After 01 & 02
└── 05_Deployment/                  ← Serving AI apps to production
```

---

## 📁 01 — Data Science Foundations

> **Recommended order:** NumPy → Pandas → EDA → SQLite

| File | Topic | Placeholder | Key Focus |
|------|-------|-------------|-----------|
| [numpy.md](01_Data_Science_Foundations/numpy.md) | NumPy | `[NUMPY SECTION]` | Arrays, broadcasting, vectorization, memory |
| [pandas.md](01_Data_Science_Foundations/pandas.md) | Pandas | `[PANDAS SECTION]` | DataFrames, cleaning, feature engineering, CoW |
| [eda.md](01_Data_Science_Foundations/eda.md) | Exploratory Data Analysis | `[EDA SECTION]` | Statistical analysis, univariate/bivariate, actionable insights |
| [sqlite.md](01_Data_Science_Foundations/sqlite.md) | SQLite | `[SQLITE SECTION]` | SQL, Python sqlite3, WAL, agent memory backends |

---

## 📁 02 — Machine Learning

> **Recommended order:** Scikit-Learn → PyTorch → Deep Learning (after completing 01)

| File | Topic | Placeholder | Key Focus |
|------|-------|-------------|-----------|
| [scikit-learn.md](02_Machine_Learning/scikit-learn.md) | Scikit-Learn | `[SKLEARN SECTION]` | Estimator API, algorithms, Pipeline, evaluation, tuning |
| [pytorch.md](02_Machine_Learning/pytorch.md) | PyTorch | `[PYTORCH SECTION]` | Tensors, autograd, nn.Module, training loop, GPU, fine-tuning |
| [deep_learning.md](02_Machine_Learning/deep_learning.md) | Deep Learning | `[DL SECTION]` | MLP, CNN, RNN, LSTM, Attention, Transformer architecture & training dynamics |

---

## 📁 03 — Data Visualization

> **Recommended order:** Matplotlib → Seaborn → Plotly → Data Visualization Principles
> *(Can be studied in parallel with 01 & 02)*

| File | Topic | Placeholder | Key Focus |
|------|-------|-------------|-----------|
| [matplotlib.md](03_Data_Visualization/matplotlib.md) | Matplotlib | `[MATPLOTLIB SECTION]` | OO-style vs pyplot, Figure/Axes/Artist hierarchy |
| [seaborn.md](03_Data_Visualization/seaborn.md) | Seaborn | `[SEABORN SECTION]` | Statistical plots, Figure-level vs Axes-level, FacetGrid |
| [plotly.md](03_Data_Visualization/plotly.md) | Plotly | `[PLOTLY SECTION]` | Interactive charts, px vs go, JSON figure structure |
| [data_visualization.md](03_Data_Visualization/data_visualization.md) | Data Visualization | `[DATA VIZ SECTION]` | Design principles, visual perception, chart selection |

---

## 📁 04 — Generative AI and LLMs

> **Recommended order:**
> LLM → Transformers → Embeddings → Vector Database → ChromaDB → RAG → Prompt Engineering → OpenAI API → Claude API → LangChain → LangGraph → LangSmith → Pydantic

| File | Topic | Placeholder | Key Focus |
|------|-------|-------------|-----------|
| [llm.md](04_Generative_AI_and_LLMs/llm.md) | Large Language Models | `[LLM SECTION]` | Transformer architecture, pre-training, RLHF, inference |
| [transformers.md](04_Generative_AI_and_LLMs/transformers.md) | HuggingFace Transformers | `[TRANSFORMERS SECTION]` | Auto classes, tokenization, generation, fine-tuning, PEFT |
| [embeddings.md](04_Generative_AI_and_LLMs/embeddings.md) | Embeddings | `[EMBEDDING SECTION]` | Vector spaces, cosine similarity, SBERT, OpenAI embeddings |
| [vector_database.md](04_Generative_AI_and_LLMs/vector_database.md) | Vector Databases | `[VECTOR DB SECTION]` | HNSW, IVF, PQ, hybrid search, DB comparison |
| [chromadb.md](04_Generative_AI_and_LLMs/chromadb.md) | ChromaDB | `[CHROMADB SECTION]` | Collections, metadata filtering, HNSW, LangChain RAG |
| [rag.md](04_Generative_AI_and_LLMs/rag.md) | RAG | `[RAG SECTION]` | Chunking, retrieval, re-ranking, evaluation, agentic RAG |
| [prompt_engineering.md](04_Generative_AI_and_LLMs/prompt_engineering.md) | Prompt Engineering | `[PROMPT ENG SECTION]` | CoT, ReAct, structured output, injection defense |
| [openai_api.md](04_Generative_AI_and_LLMs/openai_api.md) | OpenAI API | `[OPENAI SECTION]` | Chat Completions, tool calling, structured output, embeddings, fine-tuning, cost |
| [claude_api.md](04_Generative_AI_and_LLMs/claude_api.md) | Claude API (Anthropic) | `[CLAUDE SECTION]` | Messages API, Extended Thinking, Tool Use, Prompt Caching, XML prompting |
| [langchain.md](04_Generative_AI_and_LLMs/langchain.md) | LangChain | `[LANGCHAIN SECTION]` | LCEL, Runnable, chains, memory, retrievers |
| [langgraph.md](04_Generative_AI_and_LLMs/langgraph.md) | LangGraph | `[LANGGRAPH SECTION]` | Stateful graphs, agents, checkpointers, multi-agent |
| [langsmith.md](04_Generative_AI_and_LLMs/langsmith.md) | LangSmith | `[LANGSMITH SECTION]` | Tracing, evaluation, monitoring, LLM-as-a-judge |
| [pydantic.md](04_Generative_AI_and_LLMs/pydantic.md) | Pydantic v2 | `[PYDANTIC SECTION]` | Validation, Field constraints, structured LLM output |

---

## 📁 05 — Deployment

> **Recommended order:** FastAPI → Docker (after completing 04)

| File | Topic | Placeholder | Key Focus |
|------|-------|-------------|-----------|
| [fastapi.md](05_Deployment/fastapi.md) | FastAPI | `[FASTAPI SECTION]` | ASGI, async, Pydantic, Depends, streaming LLM, GenAI API patterns |
| [docker.md](05_Deployment/docker.md) | Docker | `[DOCKER SECTION]` | Containers, Dockerfile, layer caching, Compose, GPU, AI/ML stacks |

---

## 🗺️ Full Learning Roadmap

```
Phase 1 — Foundations
    NumPy → Pandas → EDA → SQLite

Phase 2 — Machine Learning
    Scikit-Learn → PyTorch → Deep Learning

Phase 3 — Visualization (parallel to Phase 1 & 2)
    Matplotlib → Seaborn → Plotly → Data Visualization

Phase 4 — Generative AI
    LLMs → HuggingFace Transformers → Embeddings
        → Vector Databases → ChromaDB
        → RAG → Prompt Engineering
        → LangChain → LangGraph → LangSmith
        → Pydantic

Phase 5 — Deployment
    FastAPI → Docker
```

---

## 📌 Quick Placeholder Reference

| Placeholder | File |
|-------------|------|
| `[NUMPY SECTION]` | numpy.md |
| `[PANDAS SECTION]` | pandas.md |
| `[EDA SECTION]` | eda.md |
| `[SQLITE SECTION]` | sqlite.md |
| `[SKLEARN SECTION]` | scikit-learn.md |
| `[PYTORCH SECTION]` | pytorch.md |
| `[DL SECTION]` | deep_learning.md |
| `[MATPLOTLIB SECTION]` | matplotlib.md |
| `[SEABORN SECTION]` | seaborn.md |
| `[PLOTLY SECTION]` | plotly.md |
| `[DATA VIZ SECTION]` | data_visualization.md |
| `[LLM SECTION]` | llm.md |
| `[TRANSFORMERS SECTION]` | transformers.md |
| `[EMBEDDING SECTION]` | embeddings.md |
| `[VECTOR DB SECTION]` | vector_database.md |
| `[CHROMADB SECTION]` | chromadb.md |
| `[RAG SECTION]` | rag.md |
| `[PROMPT ENG SECTION]` | prompt_engineering.md |
| `[OPENAI SECTION]` | openai_api.md |
| `[CLAUDE SECTION]` | claude_api.md |
| `[LANGCHAIN SECTION]` | langchain.md |
| `[LANGGRAPH SECTION]` | langgraph.md |
| `[LANGSMITH SECTION]` | langsmith.md |
| `[PYDANTIC SECTION]` | pydantic.md |
| `[FASTAPI SECTION]` | fastapi.md |
| `[DOCKER SECTION]` | docker.md |
