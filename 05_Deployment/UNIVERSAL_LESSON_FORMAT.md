# Universal Streamlit Lesson File Standard & Architecture Specification
### Single Source of Truth & Master Format for All `.py` Lesson Files in `05_Deployment`
### Reference Implementation Standard: Continuous Scroll · Graphviz Architecture · Zero LaTeX Math
### Target Domains: 1. FastAPI · 2. Streamlit · 3. Docker

---

## 1. Architectural Philosophy & Master Design Rules

Every `.py` lesson file in `05_Deployment` represents an interactive, production-grade learning module covering web service microarchitectures, API contracts, container virtualization, and production deployment lifecycles.

### 1.1 Core Principles
1. **Continuous Vertical Scroll Flow**: All content resides on a single scrollable page (`st.set_page_config(layout="wide")`). Top-level `st.tabs` are **strictly forbidden**.
2. **Zero LaTeX Math Notation**: Never use `$$...$$` or `$x$`. Express all latency percentiles, throughput calculations, and complexity metrics in plain English and code formulas (e.g., `Throughput: Requests / Second`, `P99 Latency < 50ms`, `O(1) Route Matching`).
3. **Heavy Architectural & Microservice Diagrams**: Every lesson must feature Graphviz diagrams with dark theme styling (`bgcolor="#0B1120"`):
   - **Service Mesh**: Reverse Proxy (Nginx) -> ASGI Workers (Uvicorn) -> FastAPI Application -> Database / Storage Cylinders.
   - **Container Architecture**: Base image layers -> Build stages (Multi-stage) -> Final lightweight runtime container.
   - **Lifecycle Flows**: Client Request -> Middleware Pipeline -> Dependency Injection -> Endpoint Logic -> Response Serialization.
4. **Graphviz XML Rule**: Text inside HTML-like labels (`< ... >`) is parsed by a strict XML parser. Never use raw `&`, `<`, or `>` in headers or cells (always use `and` or `&amp;`).
5. **Dual Theory & Practical Delivery**: Combine microservice architecture diagrams, production code (`FastAPI`, `Dockerfiles`, `docker-compose`), curl / HTTP request simulations, and an interactive endpoint workbench simulating API payloads, status codes, and latency budgets.
6. **Uniform Pedagogical Endings**:
   - **Section N+1: Technical Interview Preparation & Production Traps** (Exactly 5 Q&As + 4 Traps [2 error, 2 warning] + 4 Flashcards).
   - **Section N+2: Self-Assessment Checklist** (Exactly 6 Checkboxes + `st.balloons()` completion banner).

---

## 2. Complete File Anatomy (Top-to-Bottom Structure)

```
┌────────────────────────────────────────────────────────────────────────┐
│ [A] Standard Imports Block (Exactly 5 Standard Imports)                │
├────────────────────────────────────────────────────────────────────────┤
│ [B] st.set_page_config(page_title=..., page_icon=..., layout="wide")   │
├────────────────────────────────────────────────────────────────────────┤
│ [C] Title Block: st.title + st.caption + Narrative with 5-Point List   │
├────────────────────────────────────────────────────────────────────────┤
│ [D] First st.divider()                                                 │
├────────────────────────────────────────────────────────────────────────┤
│ [E] Content Sections 1 through N (Separated by st.divider())           │
│     ├── Type 1: Microservice / Deployment Infrastructure Diagram       │
│     ├── Type 2: Architectural Comparative Matrix (Monolith vs Micro)   │
│     ├── Type 3: Production Runnable FastAPI / Docker Configuration     │
│     ├── Type 4: Interactive Endpoint & Payload Request Simulator       │
│     └── Type 6: HTTP Status Codes & Latency SLA DataFrames             │
├────────────────────────────────────────────────────────────────────────┤
│ [F] Section N+1: Technical Interview Preparation (Mandatory 2nd-Last)  │
│     ├── Exactly 5 High-Frequency Q&A Expanders                         │
│     ├── Exactly 4 Production Traps (2x st.error + 2x st.warning)       │
│     └── Exactly 4 Interactive Flashcards in 2 Columns                  │
├────────────────────────────────────────────────────────────────────────┤
│ [G] st.divider()                                                       │
├────────────────────────────────────────────────────────────────────────┤
│ [H] Section N+2: Self-Assessment Checklist (Mandatory Last Section)    │
│     ├── Exactly 6 Checkboxes (All Initialized with value=True)         │
│     └── if chk1..chk6: -> st.balloons() + st.success(Next Lesson)      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Structural Component Specifications

### 3.1 Standard Imports Block
```python
import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional
```
Strictly prohibit `import os` and `from pathlib import Path`.

### 3.2 Page Configuration (`st.set_page_config`)
```python
st.set_page_config(
    page_title="<X.Y> <Lesson Title> - Deployment & Production",
    page_icon="🚀",
    layout="wide"
)
```

### 3.3 Title Block & Opening Narrative
```python
st.title("<X.Y> <Full Descriptive Lesson Title>")
st.caption("Chapter <N>: <Chapter Name> | Domain: 5. Deployment & Production")
st.markdown("""
<2–4 sentence high-level narrative detailing production serving, scalability constraints, 
container isolation, and how this architecture delivers high availability.>

In this lesson, you will master:
1. <Microservice or container component architecture 1>
2. <Request lifecycle, concurrency, or async event loop 2>
3. <Schema validation or container layering mechanism 3>
4. <Production code pattern and deployment configuration 4>
5. <Failure modes, health checks, and security isolation 5>
""")
st.divider()
```

---

## 4. Heavy Multi-Tier & Microservice Diagrams Standard

### 4.1 Production Microservice Architecture Template
```python
dot_deploy_architecture = """
digraph DeploymentArch {
    rankdir=LR;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    subgraph cluster_ingress {
        label="Edge Ingress Tier";
        bgcolor="#1E293B";
        fontcolor="#38BDF8";
        color="#334155";
        client [label="Client Application\\n(Web / Mobile / CLI)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        proxy [label="Nginx / Traefik\\n(TLS, Rate Limiting)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
    }

    subgraph cluster_compute {
        label="Application Container (Docker)";
        bgcolor="#1E293B";
        fontcolor="#10B981";
        color="#334155";
        uvicorn [label="Uvicorn ASGI Server\\n(Worker Process Pool)", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
        fastapi [label="FastAPI App Core\\n(Router + Pydantic Engine)", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
    }

    subgraph cluster_persistence {
        label="Storage & Feature Tier";
        bgcolor="#1E293B";
        fontcolor="#EC4899";
        color="#334155";
        db [label="Database / Model Storage\\n(Artifacts & Persistent State)", shape=cylinder, fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
    }

    client -> proxy [label="HTTPS / REST"];
    proxy -> uvicorn [label="Reverse Proxy"];
    uvicorn -> fastapi [label="ASGI Events"];
    fastapi -> db [label="State / Model Load"];
}
"""
st.graphviz_chart(dot_deploy_architecture, use_container_width=True)
```

---

## 5. Mandatory Pedagogical Endings

### 5.1 Technical Interview Preparation (Section N+1)
Must contain:
1. **Exactly 5 Expanders**: High-frequency deployment, ASGI concurrency, Docker layering, and microservice questions.
2. **Exactly 4 Traps**:
   - `Trap 1` (`st.error`): Blocking I/O inside `async def` endpoints (starving Uvicorn's event loop).
   - `Trap 2` (`st.warning`): Massive Docker image bloat (failing to use multi-stage builds or caching).
   - `Trap 3` (`st.error`): Running production containers as root user without security context.
   - `Trap 4` (`st.warning`): Memory leak from uncollected global state in long-lived workers.
3. **Exactly 4 Flashcards**: Rendered in a 2-column layout (`col_fc1`, `col_fc2`).

### 5.2 Self-Assessment Checklist (Section N+2)
Must contain:
- Exactly 6 checkboxes initialized with `value=True`.
- `if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:` triggering `st.balloons()` and `st.success()`.
