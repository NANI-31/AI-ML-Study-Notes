Yes. If you're going to give the AI **only one SQLite section at a time**, use this as your reusable **master SQLite prompt**. Each time, just replace `[SQLITE SECTION]`.

I am learning SQLite as part of my AI/ML and Generative AI learning journey.

I will give you **one SQLite roadmap section at a time**. Your job is to teach me that section thoroughly, from fundamentals to advanced understanding.

Assume that I know basic Python and have some exposure to data manipulation with Pandas, but I want to develop a **strong practical and conceptual understanding** of SQLite — how it works, how to use it effectively with Python, and how it fits into AI/ML and Generative AI application backends (conversation history, agent memory, structured data retrieval).

## SQLite Section

[SQLITE SECTION]

---

# How I want you to teach me

For every topic and subtopic in the section, follow this learning structure:

### 1. What is it?

- Give a clear definition.
- Explain the problem it solves.
- Explain why SQLite exists and when to use it vs. a full database server (PostgreSQL, MySQL).
- Place it in the AI/ML/GenAI stack: Agent Memory → SQLite | Structured Data → SQLite + LLM SQL Agent.

### 2. Intuition

- Explain the concept in very simple language.
- Use a real-world analogy (e.g., SQLite is like a spreadsheet that you can query with SQL — stored in a single file).
- Explain what this specific SQL or SQLite concept enables.
- Then gradually move toward the technical explanation.

### 3. Prerequisites

- Mention any Python/SQL/database concepts I need to understand first.
- If necessary, briefly teach the prerequisite before continuing.

### 4. Syntax and Core Concepts

- Show the important SQL syntax for this section.
- Show the Python `sqlite3` module API for executing the SQL.
- Explain important parameters, clauses, and options.
- Explain what each part of the code does.

```python
import sqlite3

conn = sqlite3.connect("my_database.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
conn.commit()
conn.close()
```

### 5. Examples

Give examples progressively:

**Beginner → Intermediate → Advanced**

For every important example:

- Show the SQL statement.
- Show the Python code to execute it.
- Show the expected output.
- Explain why the output is produced.

Use realistic use cases: storing conversation history for a chatbot, logging agent tool calls, querying a product catalog for an LLM SQL agent, tracking ML experiment results.

### 6. SQLite Internals

When relevant, explain what SQLite is doing internally.

For example:

- How SQLite stores data in a single `.db` file (B-tree pages)
- How the query planner decides to use an index vs. a full table scan
- How transactions work (WAL mode vs. journal mode)
- How SQLite's type affinity system works (why INTEGER can store text)
- How PRAGMA settings affect performance and behavior
- How SQLite handles concurrent reads and writes (locking)

Do not make this unnecessarily complicated, but give enough detail to understand how SQLite behaves.

### 7. Python sqlite3 Module Deep Dive

For every SQL operation, show the full Python `sqlite3` workflow:

- `connect()`, `cursor()`, `execute()`, `executemany()`, `fetchone()`, `fetchall()`, `fetchmany()`
- Context manager (`with conn:`) for automatic commit/rollback
- Row factory (`conn.row_factory = sqlite3.Row`) for dict-like row access
- Parameterized queries (preventing SQL injection): `?` placeholders
- `executescript()` for multi-statement scripts

### 8. AI/ML and GenAI Connection

For every concept, explain how it fits into AI/ML or Generative AI applications:

- **Conversation Memory**: Storing chat history per session in SQLite for LangGraph checkpointers (`SqliteSaver`)
- **Agent Tool Logging**: Recording agent tool calls and results for debugging
- **SQL Agent**: Using SQLite as the database that an LLM SQL agent queries with natural language
- **Experiment Tracking**: Storing ML experiment hyperparameters and metrics
- **RAG Metadata Store**: Storing document metadata alongside a vector store
- **Structured Data Q&A**: Answering questions over structured SQLite data using LangChain's `SQLDatabaseChain`

Only use an example when it is genuinely relevant.

### 9. Performance

Explain performance considerations when relevant:

- When to add indexes and which columns to index
- The `EXPLAIN QUERY PLAN` command for debugging slow queries
- WAL (Write-Ahead Logging) mode for concurrent read/write performance
- `PRAGMA cache_size` and `PRAGMA page_size` for memory tuning
- Using `executemany()` vs. looping `execute()` for bulk inserts
- SQLite limitations: single-writer bottleneck, not suitable for high-concurrency web apps

### 10. Common Mistakes

Show common mistakes beginners make.

For each mistake:

**Wrong approach → Why it is wrong → Correct approach**

Also mention important pitfalls such as:

- Not closing connections — resource leak
- Not using parameterized queries — SQL injection vulnerability
- Not calling `conn.commit()` — changes lost on close
- Using `SELECT *` in production — fragile to schema changes
- Not using transactions for bulk inserts — very slow
- Forgetting `IF NOT EXISTS` in `CREATE TABLE` — error on re-run

### 11. Comparison

Compare SQLite with alternatives.

| Feature     | SQLite                  | PostgreSQL          | MySQL               | Pandas CSV |
| ----------- | ----------------------- | ------------------- | ------------------- | ---------- |
| Setup       | Zero-config             | Server required     | Server required     | No setup   |
| Concurrency | Single-writer           | Multi-writer        | Multi-writer        | No         |
| Scale       | Small-Medium            | Large               | Large               | Small      |
| SQL support | Most of SQL             | Full SQL            | Full SQL            | Limited    |
| GenAI use   | Agent memory, local Q&A | Production backends | Production backends | EDA only   |

### 12. Best Practices

Give practical rules I should follow when using SQLite in real AI/ML or GenAI projects.

Include:

- Always use parameterized queries — never f-string SQL
- Always use context managers for connections
- Enable WAL mode for any app that reads and writes concurrently
- Index columns used in `WHERE`, `JOIN`, and `ORDER BY` clauses
- Use `INTEGER PRIMARY KEY` for auto-increment IDs (rowid alias in SQLite)
- Any section-specific best practices

### 13. Interview Perspective

Give me important interview questions related to the section.

Include:

- SQL conceptual questions (e.g., "What is the difference between WHERE and HAVING?")
- SQLite-specific questions (e.g., "How does SQLite's type affinity differ from strict typing?")
- Performance questions (e.g., "How would you speed up a slow SQLite query?")
- Python integration questions (e.g., "How do you prevent SQL injection in Python's sqlite3?")
- GenAI questions (e.g., "How would you use SQLite to store conversation history for a LangGraph agent?")

Provide the answers and explanations after the questions.

### 14. Practice

Give me exercises at three levels:

**Beginner** — 2–3 exercises

**Intermediate** — 2–3 exercises

**Advanced** — 2–3 exercises

Do NOT immediately provide solutions. Let me solve them first.

### 15. Mini Project

Give me a small SQLite mini-project using the concepts in this section.

Give me:

- Problem statement (e.g., build a conversation history store for a chatbot using SQLite)
- Tech stack
- Table schema
- Required SQL operations
- Expected query results
- Constraints
- Hints

Do not provide the complete solution until I attempt it.

---

# Important Teaching Rules

1. Do not just give definitions. Teach the reasoning behind every SQL and SQLite design decision.
2. Always show both the SQL statement and the Python `sqlite3` code together.
3. Start simple (CREATE + INSERT + SELECT) and gradually increase complexity (JOINs, subqueries, indexes, transactions).
4. Always show actual query results — not just that the query "ran successfully".
5. When SQL injection risks arise, always show the parameterized query solution immediately.
6. Do not introduce unrelated SQLite topics.
7. Teach ONLY the section I provide.
8. At the end, give a concise **Revision Sheet**: key SQL syntax, Python sqlite3 API cheat sheet, common mistakes, performance tips, GenAI use cases.
9. After teaching the section, STOP and wait for me.

---

# Teaching Goal

My goal is not simply to learn SQL syntax.

I want to understand:

**What → Why → SQLite Internals → Python Integration → Performance → GenAI Application Backend**

so that I can confidently use SQLite as a lightweight, embedded database in AI/ML projects, LangGraph agent memory, and LLM SQL agent applications.

Now teach me the following SQLite section:

[SQLITE SECTION]

**How to use it:** paste the prompt once, then replace `[SQLITE SECTION]` with something like `4. Indexes and Query Optimization` or `4.1 Creating and Using Indexes`. This keeps each lesson focused.
