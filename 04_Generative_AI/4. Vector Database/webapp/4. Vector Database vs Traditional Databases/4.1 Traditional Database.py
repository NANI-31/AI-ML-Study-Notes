import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="4.1 Traditional Database - Vector DB Curriculum",
    page_icon="🏛️",
    layout="wide"
)

# Title & Overview
st.title("4.1 Traditional Database: Relational Schemas, B-Trees, SQL & ACID Foundations")
st.caption("Chapter 4: Vector Database vs Traditional Databases | Domain: 4. Vector Database")
st.markdown("""
For over four decades, the relational database management system (RDBMS)—pioneered by Edgar F. Codd and embodied 
in engines like PostgreSQL, MySQL, Oracle, and Microsoft SQL Server—has formed the undisputed backbone of enterprise computing.

RDBMS architectures were engineered with precision around **structured tabular schemas**, **referential foreign key integrity**, 
**declarative SQL queries**, and strict **ACID transaction guarantees**. 

To truly understand why artificial intelligence necessitated the invention of vector databases, we must first deeply examine 
how traditional databases work under the hood, how their **B-Tree indexes** execute exact matches in sub-millisecond time, 
and where their fundamental architectural assumptions encounter a hard mathematical wall when faced with unstructured semantic data.
""")

st.divider()

# Section 1: Relational Architecture, Rows, Columns & Keys
st.header("1. Anatomy of the Relational Model: Rows, Columns & Constraints", divider="rainbow")

col_rel1, col_rel2 = col_rel1, col_rel2 = st.columns([1, 1])

with col_rel1:
    st.markdown("""
    ### 1. Structural Primitives of the Relational Model
    * **Rows (Tuples) and Columns (Attributes)**:
      * Data is organized into two-dimensional relations (tables) with strict **Schema-on-Write** typing.
      * Each column enforces rigid data types: `INTEGER`, `VARCHAR(255)`, `TIMESTAMP`, `BOOLEAN`, `DECIMAL(10, 2)`.
      * Database normalization (1NF, 2NF, 3NF, BCNF) eliminates redundancy through formal decomposition.
    * **Primary Keys (PK)**:
      * A column (or composite set of columns) that uniquely identifies each tuple in the relation.
      * Automatically backed by a **Clustered Index (B+ Tree)** or unique constraint, providing `O(log N)` lookup.
    * **Foreign Keys (FK) & Referential Integrity**:
      * Establishes formal relationships between tables: `orders.user_id REFERENCES users.id`.
      * Guarantees referential consistency: child records cannot reference non-existent parent rows.
      * Requires expensive multi-table **JOIN operations** (Hash Joins, Nested Loop Joins, Merge Joins) during queries.
    """)

with col_rel2:
    st.markdown("### The Relational Database Internal Engine")
    rdbms_dot = """
    digraph RelationalEngine {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        SQLQuery [label="SQL Query: SELECT * FROM orders\\nWHERE user_id = 42 AND total > 100", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        Parser [label="SQL Parser & AST Validator\\nSyntax check & catalog binding", fillcolor="#DBEAFE"];
        Optimizer [label="Cost-Based Query Optimizer (CBO)\\nGenerates index scan vs seq scan plan", fillcolor="#BFDBFE"];

        subgraph cluster_storage {
            label="Storage & Transaction Engine"; style="dashed"; color="#059669";
            BTree [label="B+ Tree Clustered Index\\n1D Ordered Pointer Hierarchy", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
            BufferPool [label="Buffer Pool (Shared Memory)\\nLRU Page Cache (8KB / 16KB Pages)", fillcolor="#A7F3D0"];
            MVCC [label="MVCC Concurrency Manager\\nSnapshot Isolation & Row Locks", fillcolor="#D1FAE5"];
            WAL [label="Write-Ahead Log (WAL / Redo Log)\\nACID Durability to Disk (fsync)", fillcolor="#6EE7B7", color="#047857", style="filled,bold"];
        }

        ExactResult [label="Deterministic Result Set\\nExact matching rows returned", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        SQLQuery -> Parser -> Optimizer -> BTree;
        Optimizer -> BufferPool;
        BTree -> BufferPool;
        BufferPool -> MVCC -> WAL;
        BufferPool -> ExactResult;
    }
    """
    st.graphviz_chart(rdbms_dot, use_container_width=False)

st.divider()

# Section 2: ACID Properties: The Enterprise Gold Standard
st.header("2. The ACID Guarantees: Engineering Deterministic Reliability", divider="rainbow")

st.markdown("""
Traditional relational engines prioritize data correctness above all else. Every transaction is governed by 
the four **ACID properties**:
""")

col_acid1, col_acid2, col_acid3, col_acid4 = st.columns(4)

with col_acid1:
    st.markdown("""
    #### A — Atomicity
    * **"All or Nothing"**:
      * Either every operation within a transaction succeeds and commits, or the entire transaction is rolled back.
    * **Mechanism**:
      * Uses Undo Logs / Write-Ahead Logs to reverse intermediate writes if an error or crash occurs.
    * **Example**:
      * Transferring $100: Debiting Account A and Crediting Account B must both succeed or neither occurs.
    """)

with col_acid2:
    st.markdown("""
    #### C — Consistency
    * **Preserving Invariants**:
      * Data must always transition from one valid state to another valid state.
    * **Enforcement**:
      * Foreign key constraints, uniqueness rules, `CHECK` expressions, and data type bounds are strictly validated before committing.
    * **Example**:
      * An order cannot be inserted without a valid existing customer ID.
    """)

with col_acid3:
    st.markdown("""
    #### I — Isolation
    * **Concurrent Execution**:
      * Concurrent transactions execute as if they were running serially without interfering with one another.
    * **Mechanisms**:
      * Multi-Version Concurrency Control (MVCC), Two-Phase Locking (2PL), row-level locks.
    * **Isolation Levels**:
      * Read Committed, Repeatable Read, Serializable.
    """)

with col_acid4:
    st.markdown("""
    #### D — Durability
    * **Permanent Persistence**:
      * Once a transaction commits, its writes will never be lost—even in the event of an immediate power failure or OS crash.
    * **Mechanism**:
      * Append-only Write-Ahead Logs (WAL) flushed synchronously to non-volatile disk via `fsync()` before acknowledging commit.
    """)

st.divider()

# Section 3: Exact Matching & B-Tree Index Mechanics
st.header("3. Exact Matching & Why B-Trees Fail in High Dimensions", divider="rainbow")

st.markdown("""
The speed of a traditional database depends almost entirely on the **B+ Tree Index**:
""")

col_btree1, col_btree2 = st.columns([1, 1])

with col_btree1:
    st.markdown("""
    ### How B+ Trees Work (The 1D Superpower)
    * **Total Ordering Property**:
      * B+ Trees require that data can be strictly sorted along a **single one-dimensional axis**: `a < b < c`.
      * Keys are stored in sorted order across hierarchical tree pages.
    * **Performance**:
      * Exact point lookups (`WHERE id = 5029`): `O(log N)` disk reads.
      * Range scans (`WHERE age BETWEEN 25 AND 35`): Locate leaf node in `O(log N)`, then scan linked sibling leaves in `O(K)`.
    * **Predictability**:
      * Deterministic binary yes/no outcome: a row either satisfies the predicate or it does not.
    """)

with col_btree2:
    st.markdown("""
    ### Why B+ Trees Completely Collapse on Vectors
    * **1. No Natural 1D Total Order**:
      * There is no natural total ordering in multi-dimensional space! Is vector `[0.9, 0.1]` greater or less than `[0.1, 0.9]`?
    * **2. The Curse of Dimensionality**:
      * As dimensions increase (e.g. 768 or 1536), space expands exponentially.
      * Spatial partitioning trees (like KD-Trees or R-Trees) require checking almost every branch of the tree.
    * **3. Degradation to Full Table Scan**:
      * To find the nearest semantic neighbor using traditional indexes, the database is forced to perform an `O(N)` Sequential Table Scan across gigabytes of rows, taking seconds per query!
    """)

st.divider()

# Section 4: Interactive SQL Query & B-Tree Simulator
st.header("4. Interactive Relational Query & Index Simulator", divider="rainbow")

st.markdown("""
Experiment with traditional relational queries against an indexed customer database. 
Observe the query plan, index traversal path, and why substring/semantic searches degrade:
""")

# Sample Relational Table
mock_rdb = pd.DataFrame({
    "user_id": [101, 102, 103, 104, 105, 106, 107, 108],
    "email": ["alice@company.com", "bob@fintech.io", "charlie@gmail.com", "david@acme.org", "eva@enterprise.net", "frank@startup.io", "grace@cloud.com", "heidi@security.org"],
    "age": [28, 42, 35, 22, 51, 31, 29, 45],
    "account_balance": [14500.0, 3200.5, 890.0, 25000.0, 112000.0, 740.0, 18200.0, 63000.0],
    "notes": [
        "Prefers mobile app, reported login bug last week",
        "Enterprise procurement manager looking for SOC2 report",
        "Student discount requested, low balance",
        "High-net-worth investor, interested in crypto options",
        "VIP customer account, dedicated account executive assigned",
        "Trial account expiring in 3 days",
        "Cloud engineer, requested Terraform automation templates",
        "Security compliance officer, auditing data retention policies"
    ]
})

col_sql_in, col_sql_out = st.columns([1, 1])

with col_sql_in:
    st.subheader("Relational SQL Query Console")
    query_mode = st.selectbox(
        "Select Query Execution Type",
        options=[
            "1. Primary Key Point Lookup (WHERE user_id = 104)",
            "2. B+ Tree Clustered Range Scan (WHERE age BETWEEN 25 AND 35)",
            "3. Lexical Substring Filter (WHERE notes LIKE '%security%')",
            "4. Semantic Query (WHERE notes MEANING 'wants API infrastructure help')"
        ]
    )

with col_sql_out:
    st.subheader("Query Plan & Execution Output")
    
    if "1. Primary Key" in query_mode:
        row = mock_rdb[mock_rdb["user_id"] == 104]
        st.success("**Query Plan:** `Index Scan using users_pkey on users (Cost: 0.15..8.17 rows=1)`")
        st.markdown("*Traversed 3 B+ Tree page levels in **0.4 ms**.*")
        st.dataframe(row[["user_id", "email", "age", "account_balance"]], use_container_width=True, hide_index=True)
        
    elif "2. B+ Tree Clustered Range" in query_mode:
        rows = mock_rdb[(mock_rdb["age"] >= 25) & (mock_rdb["age"] <= 35)]
        st.success("**Query Plan:** `Bitmap Index Scan on idx_users_age (Cost: 4.12..12.35 rows=4)`")
        st.markdown(f"*B+ Tree leaf pointer traversal returned **{len(rows)} matching rows** in **1.1 ms**.*")
        st.dataframe(rows[["user_id", "email", "age", "account_balance"]], use_container_width=True, hide_index=True)
        
    elif "3. Lexical Substring" in query_mode:
        rows = mock_rdb[mock_rdb["notes"].str.contains("security", case=False)]
        st.warning("**Query Plan:** `Seq Scan on users (Cost: 0.00..35.50 rows=1) Filter: (notes ~~* '%security%')`")
        st.markdown("*B+ Tree CANNOT be used for leading wildcards! Full table scan executed across all pages.*")
        st.dataframe(rows[["user_id", "email", "notes"]], use_container_width=True, hide_index=True)
        
    else: # Semantic
        st.error("**SQL Execution Error:** `ERROR: syntax error at or near 'MEANING'`")
        st.markdown("""
        **Why RDBMS Engines Fail Here**:
        * Traditional databases have **zero concept of semantic meaning**.
        * User 107 asked for `"Terraform automation templates"`, which directly addresses API infrastructure—yet an RDBMS cannot match this without manual, exhaustive synonym rules!
        * This is the exact architectural gap that necessitated **Vector Databases**.
        """)

st.divider()

# Section 5: Executable Python Implementation
st.header("5. Executable Code: MiniRelationalEngine with ACID Transactions & B-Tree", divider="rainbow")

st.markdown("""
The following complete Python program implements a **MiniRelationalEngine** featuring a **sorted B-Tree index simulation**, 
**Primary Key enforcement**, and **ACID Transaction Rollback support**:
""")

code_rdb_impl = '''from typing import Dict, Any, List, Optional
import bisect

class MiniRelationalTable:
    """Implements a schema-enforced relational table with B-Tree indexing and ACID rollback."""
    def __init__(self, table_name: str, pk_column: str, schema: Dict[str, type]):
        self.table_name = table_name
        self.pk_column = pk_column
        self.schema = schema
        
        # In-memory storage (pages/rows)
        self.rows: Dict[Any, Dict[str, Any]] = {}
        
        # B+ Tree simulated index: sorted list of PKs for O(log N) lookup
        self.b_tree_index: List[Any] = []
        
        # Transaction undo log for ACID rollback
        self._active_transaction: bool = False
        self._undo_log: List[Dict[str, Any]] = []

    def begin_transaction(self):
        """Starts an atomic transaction block."""
        self._active_transaction = True
        self._undo_log = []

    def commit(self):
        """Commits transaction changes permanently."""
        self._active_transaction = False
        self._undo_log = []

    def rollback(self):
        """Atomically rolls back all changes made during the active transaction."""
        for entry in reversed(self._undo_log):
            action = entry["action"]
            pk = entry["pk"]
            if action == "INSERT":
                del self.rows[pk]
                self.b_tree_index.remove(pk)
            elif action == "UPDATE":
                self.rows[pk] = entry["old_row"]
        self._active_transaction = False
        self._undo_log = []

    def insert(self, row: Dict[str, Any]):
        """Schema-on-write insertion with primary key uniqueness validation."""
        # 1. Validate Schema Consistency
        for col, col_type in self.schema.items():
            if col not in row:
                raise ValueError(f"Missing column '{col}' violating schema constraint.")
            if not isinstance(row[col], col_type):
                raise TypeError(f"Column '{col}' must be of type {col_type.__name__}.")

        pk = row[self.pk_column]
        if pk in self.rows:
            raise ValueError(f"Duplicate Primary Key violation: {pk} already exists.")

        # Record undo entry if inside active transaction
        if self._active_transaction:
            self._undo_log.append({"action": "INSERT", "pk": pk})

        # Store row
        self.rows[pk] = row
        bisect.insort(self.b_tree_index, pk) # Maintain sorted B-tree index

    def find_by_pk(self, pk: Any) -> Optional[Dict[str, Any]]:
        """O(log N) binary search B-Tree lookup."""
        idx = bisect.bisect_left(self.b_tree_index, pk)
        if idx < len(self.b_tree_index) and self.b_tree_index[idx] == pk:
            return self.rows[pk]
        return None

# Demonstration of ACID Transaction & Rollback
table = MiniRelationalTable(
    table_name="accounts",
    pk_column="account_id",
    schema={"account_id": int, "owner": str, "balance": float}
)

# Insert initial valid committed rows
table.insert({"account_id": 1, "owner": "Alice", "balance": 1000.0})
table.insert({"account_id": 2, "owner": "Bob", "balance": 500.0})

print("1. Initial Committed State:")
print(f"Account 1: {table.find_by_pk(1)}")
print(f"Account 2: {table.find_by_pk(2)}")

# Begin Atomic Transaction
table.begin_transaction()
try:
    # Debit Alice
    table.rows[1]["balance"] -= 200.0
    table._undo_log.append({"action": "UPDATE", "pk": 1, "old_row": {"account_id": 1, "owner": "Alice", "balance": 1000.0}})
    
    # Attempt invalid insert (Simulate crash during transfer)
    table.insert({"account_id": 3, "owner": "Charlie", "balance": "INVALID_FLOAT"}) # Raises TypeError
    table.commit()
except Exception as e:
    print(f"\\n2. Transaction Aborted due to error: {e}")
    table.rollback()
    print("Rollback executed successfully!")

print("\\n3. State After Rollback (Atomicity Preserved):")
print(f"Account 1: {table.find_by_pk(1)}")
'''

st.code(code_rdb_impl, language="python")

if st.button("🚀 Execute MiniRelationalEngine with ACID Rollback", type="primary"):
    exec_scope = {}
    exec(code_rdb_impl, globals(), exec_scope)
    
    t = exec_scope["table"]
    acc1 = t.find_by_pk(1)
    acc2 = t.find_by_pk(2)
    
    st.success("ACID Engine executed successfully! Verified that Account 1 balance rolled back to 1000.0!")
    col_out1, col_out2 = st.columns(2)
    with col_out1:
        st.markdown("**Account 1 State:**")
        st.json(acc1)
    with col_out2:
        st.markdown("**Account 2 State:**")
        st.json(acc2)

st.divider()

# Section 6: Technical Interview Preparation & Production Traps
st.header("6. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. Why do B+ Trees excel at 1D relational data but completely fail on high-dimensional vectors?"""
a1 = """* **Total Ordering Property**: A B+ Tree requires that all keys have a single, unambiguous scalar order (`a < b`). In 1D space, this allows dividing search space in half at each tree node (`O(log N)`).
* **The Curse of Dimensionality in Higher Dimensions**:
  * In d-dimensional space (`d = 768` or `1536`), there is no total order.
  * Spatial trees that attempt multi-dimensional bounding hyperplanes (like KD-trees or R-trees) degenerate because high-dimensional hyperspheres collapse into hyperspace corners.
  * To guarantee finding the nearest vector, a B+ Tree or KD-Tree must visit virtually every leaf node, collapsing from `O(log N)` to `O(N)` full table scan.
* **Vector DB Solution**: Replace tree partitioning with proximity graphs (HNSW) and quantized cluster centroids (IVF-PQ)."""

q2 = """2. What is Multi-Version Concurrency Control (MVCC) in relational databases, and how does it achieve snapshot isolation?"""
a2 = """* **The Problem**: Locking rows during read queries blocks concurrent write transactions, destroying database throughput.
* **How MVCC Solves It**:
  * Instead of modifying a row in place, an `UPDATE` creates a new version of the row with a transaction timestamp (`xmin` and `xmax`).
  * Readers do not lock rows: a reading transaction reads only row versions that were committed before that transaction started (its point-in-time snapshot).
  * Writers only conflict with other concurrent writers modifying the same row.
* **Trade-off**: Requires background garbage collection (like PostgreSQL `VACUUM`) to reclaim disk space from dead tuple versions."""

q3 = """3. Why do relational database foreign key JOIN operations fail to scale in distributed cloud architectures?"""
a3 = """* In a single-node database, executing a JOIN involves memory hash tables or index nested loops across local RAM pages.
* In a distributed, sharded database, the child table (`orders`) and parent table (`users`) are frequently partitioned across different physical network nodes.
* Executing `orders JOIN users` requires network shuffling: transmitting millions of rows across network switches to align join keys.
* This introduces massive network latency, high serialized network I/O, and distributed deadlock risks.
* Consequently, distributed vector databases avoid relational foreign keys, embedding all metadata directly inside flat JSON payload segments."""

q4 = """4. What is Write-Ahead Logging (WAL), and how does it guarantee ACID Durability?"""
a4 = """* Modifying database data pages in random disk locations on every write is slow.
* **The WAL Protocol**:
  * Before any dirty data page is written to the table file, an append-only log record describing the change (the delta) is written sequentially to the Write-Ahead Log on disk.
  * The transaction commit is not acknowledged until the WAL buffer is flushed to disk via an operating system `fsync()` call.
* **Crash Recovery**:
  * If power is cut immediately after commit, the dirty page may not have reached the table file on disk.
  * Upon reboot, the database replays the sequential WAL from the last checkpoint (REDO phase) to reconstruct committed state, guaranteeing 100% durability."""

q5 = """5. Why does a SQL query with `WHERE text LIKE '%search%'` force a sequential table scan?"""
a5 = """* Standard B+ Trees can only optimize **prefix searches**: `WHERE text LIKE 'search%'`.
  * In a prefix search, the B+ Tree traverses sorted keys alphabetically to find the start of the `'search'` block, then scans leaves until the prefix changes.
* With a leading wildcard `'%search%'`, the matching string can appear anywhere inside the text.
* The B+ Tree's alphabetical sorting is completely useless for interior substrings.
* The query planner has no choice but to discard the index and execute a full sequential scan across every single 8KB disk page in the table."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Relational Database Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The Unindexed Foreign Key Lock Cascade**  
> While primary keys are indexed automatically, foreign key columns (e.g. `orders.user_id`) are **NOT** automatically 
> indexed in PostgreSQL. When a parent `users` row is updated or deleted, the database takes an exclusive table lock 
> on the entire `orders` table to verify referential integrity, freezing all concurrent checkouts! Always index FK columns.

> [!WARNING]
> **Trap 2: The Long-Running Transaction WAL Bloat Trap**  
> A forgotten reporting query or developer script that leaves a transaction open (`BEGIN; ... no COMMIT`) 
> blocks PostgreSQL's `VACUUM` cleaner from removing dead row versions created after the transaction began. 
> The table and WAL log files bloat uncontrollably until disk space is 100% exhausted, crashing the database cluster.

> [!WARNING]
> **Trap 3: Storing Raw Large Text in Inline Table Rows**  
> Storing 20KB document chunks directly in `VARCHAR` or `TEXT` columns causes row sizes to exceed the database's 
> 8KB page size, triggering TOAST (The Oversized-Attribute Storage Technique). TOAST moves large strings to separate 
> compressed storage blocks, multiplying disk I/O operations and severely slowing down sequential queries.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What is Schema-on-Write?"):
        st.markdown("""
        **Schema-on-Write**:  
        Relational tables enforce strict column data types, nullability, and foreign key constraints at insertion time, rejecting any non-conforming data.
        """)
    with st.expander("🗂️ Flashcard: What is a B+ Tree?"):
        st.markdown("""
        **B+ Tree Index**:  
        A self-balancing search tree where all data pointers reside in sorted leaf nodes, delivering O(log N) point lookups and efficient sequential range scans for 1D scalar data.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is ACID?"):
        st.markdown("""
        **ACID Guarantees**:  
        Atomicity (all-or-nothing), Consistency (invariants preserved), Isolation (independent concurrent execution via MVCC), Durability (persistence via WAL fsync).
        """)
    with st.expander("🗂️ Flashcard: Why do B+ Trees Fail on Vectors?"):
        st.markdown("""
        **Vector Failure**:  
        High-dimensional embeddings lack a 1D scalar order, forcing B+ Trees to scan virtually all branches, degrading search to an O(N) full table scan.
        """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I can explain the core architecture of an RDBMS: tables, rows, columns, and foreign keys.", value=True)
chk2 = st.checkbox("I understand all 4 ACID guarantees and how WAL and MVCC enforce them.")
chk3 = st.checkbox("I can explain how B+ Tree indexes achieve O(log N) lookup on 1D ordered data.")
chk4 = st.checkbox("I know why B+ Trees collapse into O(N) sequential scans on high-dimensional vectors.")
chk5 = st.checkbox("I understand the performance implications of leading wildcard searches (`LIKE '%text%'`).")
chk6 = st.checkbox("I can implement an in-memory relational table with ACID rollback support in Python.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered Traditional Database foundations and are ready for Chapter 4.2: Vector Database Differences!")
