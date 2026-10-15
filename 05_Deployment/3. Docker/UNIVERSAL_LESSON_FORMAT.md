# Universal Streamlit Lesson File Standard & Architecture Specification
### Single Source of Truth & Master Format for Domain: `Docker` (Containerization, Images, Volumes, Networks, Compose & Deployment)
### Target Curriculum: `05_Deployment/3. Docker/docker.txt` (Chapters 1–25)
### Core Reference Standard: Continuous Vertical Scroll · Dark Graphviz · Top-to-Bottom Flow · Zero LaTeX Math · OCI & BuildKit Core

---

## 1. Architectural Philosophy & Master Design Rules

Every `.py` lesson file in `3. Docker` represents an interactive, production-grade learning module covering containerization architectures, Linux kernel isolation primitives, Dockerfile optimization, BuildKit caching, multi-stage builds, networking topologies, persistent volumes, Docker Compose orchestration, security hardening, and production deployment pipelines.

To ensure consistency, pedagogical depth, and zero UI breakage across all lessons, **every single generated `.py` file must strictly follow this single unified structure**.

### 1.1 Core Principles
1. **Continuous Vertical Scroll Flow**: All content resides on a single scrollable page (`st.set_page_config(layout="wide")`). Top-level `st.tabs` are **strictly forbidden** because they hide educational content behind clicks. (Sub-tabs are only permitted inside self-contained widgets or workbench result panels).
2. **Mandatory 1:1 Section Density (Every Roadmap Bullet Gets a Dedicated Section)**: Every single numbered bullet point under a lesson in `docker.txt` **MUST receive its own dedicated `st.header("X. <Topic Name>", divider="rainbow")` section**. Never compress or merge topics.
   - Total Sections per lesson file is locked by formula:
     `Total Sections = 1 (Architecture Blueprint) + K (Roadmap Syllabus Topics) + 1 (Production Implementation) + 1 (Interactive Workbench) + 1 (Technical Interview Prep) + 1 (Self-Assessment Checklist) = K + 5`.
3. **Clean Single Sequential Section Numbering (Strictly Zero Compound Numbering)**: Section headers MUST ALWAYS use clean, single sequential integer numbering from 1 to N (`st.header("1. <Blueprint>", divider="rainbow")`, `st.header("2. <Topic 1 Title>", divider="rainbow")`, ..., `st.header(f"{K+2}. Production Implementation: ...")`). Strictly prohibit compound, dotted, or nested section numbering (e.g. NEVER use `st.header("2. 1. ...")`, `st.header("2.1 ...")`, `st.header("3. 2. ...")`).
4. **ABSOLUTELY ZERO MATHEMATICAL EQUATIONS**:
   - This curriculum is strictly focused on **Systems Engineering, Operating System Primitives, and Infrastructure Operations**.
   - Strictly prohibit ALL LaTeX notation (`$$...$$`, `$x$`), calculus, loss functions, and theoretical equations.
   - Frame all concepts in pure Systems Architecture: Linux kernel isolation primitives (namespaces: pid, net, ipc, mnt, uts, user; control groups: memory, cpu, blkio), UnionFS / Overlay2 copy-on-write storage layers, virtual ethernet bridges (veth), OCI image manifests, and container process lifecycles.
5. **Modern Container Production Reality Contrast**:
   - Every section covering legacy or suboptimal practices must feature a prominent callout block:
     `### 🚀 Modern Production Reality: How Current Industry Stacks Implement This`
   - Explicitly contrast legacy vs modern implementations:
     - Heavy monolithic 2GB images vs Minimal multi-stage builds with Distroless or Alpine base images.
     - Fat developer VMs ("works on my machine") vs Standardized OCI containers and reproducible Dev Containers.
     - Running processes as `root` vs Principle of least privilege with non-root runtime users and rootless Docker.
     - Naive Dockerfile ordering that invalidates build cache vs Optimized layer ordering and BuildKit cache mounts (`--mount=type=cache`).
     - Hardcoded credentials in images/ENV vs Docker Secrets, HashiCorp Vault, and runtime environment injection.
6. **Mandatory Top-to-Bottom Diagram Flow (`rankdir=TB;`) & `st.expander` Encapsulation**:
   - **Top-to-Bottom Flow (`rankdir=TB;`) Mandatory**: Long horizontal pipelines (`rankdir=LR;` with more than 3–4 sequential nodes) are **strictly forbidden** because Streamlit scales SVGs to container width, crushing wide diagrams into flat, illegible horizontal strips. All container lifecycles, build stages, and network graphs must use **`rankdir=TB;` (Top to Bottom)** so they expand vertically with readable fonts (`fontsize=11`) and generous margins (`margin="0.2,0.1"`).
   - **Mandatory `st.expander` Container for Blueprints**: Wrap macro architectural blueprints inside an expander:
     ```python
     with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
         st.graphviz_chart(dot_blueprint, use_container_width=True)
     ```
   - **Contextual In-Section Diagrams (Mandatory Whenever Required)**: Diagrams must NOT be restricted to Section 1 alone. Whenever any roadmap topic section covers a multi-step workflow, data structure, or system topology (e.g. Dockerfile layer cache stacks, container state transitions, bridge network routing, volume mount topologies, multi-stage builder-to-runtime artifact copy), it **MUST include its own dedicated contextual Graphviz diagram inside that specific section**.
7. **Graphviz XML Rule**: Text inside HTML-like labels (`< ... >`) is parsed by a strict XML parser. Never use raw `&`, `<`, or `>` in headers or cells (always use `and` or `&amp;`).
8. **Module Scope for Interactive Workbench Classes**: Any classes, Dockerfile linters, layer analyzers, or Compose mock engines needed by the Interactive Workbench must be defined in real Python module scope within the script. Never define executable classes solely inside string literals passed to `st.code(...)`.
9. **Standard Imports**: Exactly 5 standard imports at the top of the file:
   ```python
   import streamlit as st
   import pandas as pd
   import numpy as np
   import graphviz
   from typing import List, Dict, Any, Tuple, Optional
   ```
   Strictly prohibit `import os` and `from pathlib import Path`.
10. **Uniform Pedagogical Endings**: Every single lesson concludes with:
    - **Section K+4: Technical Interview Preparation & Production Traps** (Exactly 5 Q&As + 4 Traps [2 error, 2 warning] + 4 Flashcards).
    - **Section K+5: Self-Assessment Checklist** (Exactly 6 Checkboxes + clean `st.success(...)` completion banner. Strictly prohibit `st.balloons()`).

---

## 2. Complete File Anatomy (Top-to-Bottom Structure)

```
┌────────────────────────────────────────────────────────────────────────┐
│ [A] Standard Imports Block (Exactly 5 Standard Imports)                │
├────────────────────────────────────────────────────────────────────────┤
│ [B] st.set_page_config(page_title=..., page_icon=..., layout="wide")   │
├────────────────────────────────────────────────────────────────────────┤
│ [C] Title Block: st.title + st.caption + Narrative with 4-5 Pillars    │
├────────────────────────────────────────────────────────────────────────┤
│ [D] First st.divider()                                                 │
├────────────────────────────────────────────────────────────────────────┤
│ [E] Section 1: End-to-End Architectural Blueprint (Graphviz Dark TB)   │
│     └── Wrapped in st.expander with rankdir=TB;                        │
├────────────────────────────────────────────────────────────────────────┤
│ [F] Sections 2 through K+1: Mandatory 1:1 Syllabus Roadmap Topics      │
│     ├── Deep conceptual explanation (Linux Primitives / OCI Specs)     │
│     ├── 📊 Contextual Diagram (Mandatory whenever workflows appear)    │
│     ├── 🚀 Modern Production Reality: Industry Standard Callout        │
│     ├── Side-by-Side Comparison Matrix (Legacy vs Modern)              │
│     └── Interactive Mini-Demo / Code Demonstration                     │
├────────────────────────────────────────────────────────────────────────┤
│ [G] Section K+2: Production Implementation: End-to-End Pipeline        │
│     ├── Real-world runnable Python / Docker automation pipeline        │
│     └── Robust configuration, health checks & lifecycle management     │
├────────────────────────────────────────────────────────────────────────┤
│ [H] Section K+3: Interactive Workbench Simulator                       │
│     ├── Real-time input manipulation widgets (Sliders, Text Areas)     │
│     └── Live Dockerfile Linter / Cache Layer Inspector                 │
├────────────────────────────────────────────────────────────────────────┤
│ [I] Section K+4: Technical Interview Preparation & Production Traps    │
│     ├── Exactly 5 High-Frequency Interview Q&A Expanders               │
│     ├── Exactly 4 Production Traps (2x st.error, 2x st.warning)        │
│     └── Exactly 4 Flashcards in a 2-Column Responsive Grid             │
├────────────────────────────────────────────────────────────────────────┤
│ [J] st.divider()                                                       │
├────────────────────────────────────────────────────────────────────────┤
│ [K] Section K+5: Self-Assessment Checklist (Mandatory Last Section)    │
│     ├── Exactly 6 Checkboxes (All Initialized with value=True)         │
│     └── if chk1..chk6: -> st.success(...) [Strictly NO balloons]       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Structural Component Code Templates

### 3.1 Title Block & 5-Pillar Overview
```python
st.title("<X.Y> <Full Descriptive Lesson Title>")
st.caption("Chapter <N>: <Chapter Name> | Domain: Docker & Container Systems")
st.markdown("""
<2-4 sentence high-level overview explaining how this container primitive operates,
why microservices and ML systems require containerized packaging, and how it ensures production parity.>

In this lesson, you will master:
- **Core Architecture & Primitives**: How the Docker Engine and Linux isolation primitives function.
- **Image & Layer Optimization**: Caching strategies, copy-on-write storage, and minimal footprint.
- **Networking & Persistence**: Data volumes, bind mounts, and inter-container service discovery.
- **Production Hardening**: Security boundaries, non-root users, and healthcheck verification.
- **Systems Interview Readiness**: Architecture trade-offs and high-frequency technical questions.
""")
st.divider()
```

### 3.2 Dark-Theme Graphviz Template for Docker Architecture (Top-to-Bottom Flow)
```python
dot_blueprint = """
digraph DockerArchitecture {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", color="#334155", fillcolor="#1E293B", margin="0.2,0.1"];
    edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

    subgraph cluster_client {
        label="Client Tier";
        fontcolor="#38BDF8";
        style=dashed;
        color="#334155";
        docker_cli [label="Docker CLI / Compose\\n(Commands, REST API calls)", fillcolor="#0F172A"];
        dockerfile [label="Dockerfile / Source Context\\n(Declarative Spec & Assets)", fillcolor="#1E293B"];
    }

    subgraph cluster_host {
        label="Docker Host & Engine (Daemon)";
        fontcolor="#F59E0B";
        style=dashed;
        color="#334155";
        dockerd [label="Docker Daemon (dockerd)\\nAPI Server & Process Supervisor", fillcolor="#1E293B"];
        buildkit [label="BuildKit Engine\\n(Parallel Builds, Cache Mounts)", fillcolor="#1E293B"];
        containerd [label="containerd & runc\\n(OCI Runtime, Namespaces, Cgroups)", fillcolor="#1E293B"];
        storage_engine [label="Storage Driver (Overlay2)\\n(Read-Only Layers + Writable CoW)", fillcolor="#1E293B"];
    }

    subgraph cluster_runtime {
        label="Container Runtime & Networks";
        fontcolor="#10B981";
        style=dashed;
        color="#334155";
        container_app [label="Running Container\\n(Isolated Process Tree)", fillcolor="#064E3B"];
        volume_mount [label="Persistent Volume\\n(/var/lib/docker/volumes)", fillcolor="#064E3B"];
    }

    docker_cli -> dockerd [label="Unix Socket / TLS"];
    dockerfile -> buildkit [label="Build Context"];
    dockerd -> containerd [label="gRPC"];
    containerd -> storage_engine [label="Mount Overlay"];
    containerd -> container_app [label="Spawn Process"];
    volume_mount -> container_app [label="Mount (Bypass CoW)"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)
```

---

## 4. Master Specification Reference Implementation

Below is the complete reference implementation demonstrating all required sections:

```python
import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="1.1 What is Docker? - Docker Curriculum",
    page_icon="🐳",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("1.1 What is Docker?")
st.caption("Chapter 1: Introduction to Docker | Domain: Docker & Container Systems")
st.markdown("""
Docker is an open-source platform that enables developers to build, package, distribute, and run applications
as lightweight, portable containers. By packaging application code alongside its runtime dependencies and system
libraries into standardized OCI units, Docker eliminates the classic "works on my machine" failure mode across
development, staging, and production environments.

In this lesson, you will master:
- **Containerization Mechanics**: How operating system-level virtualization differs from hardware virtualization.
- **Core Advantages**: Portability, resource efficiency, millisecond startup, and environment isolation.
- **Engine Architecture**: How the Docker Client, Daemon, containerd, and runc coordinate.
- **Enterprise Use Cases**: Microservices, Machine Learning model serving, and CI/CD automation.
- **Production Traps**: Pitfalls of monolithic image sizes and running unconfined root processes.
""")
st.divider()

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Virtual Machines vs Containers", divider="rainbow")

dot_blueprint = """
digraph VMvsContainer {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", color="#334155", fillcolor="#1E293B", margin="0.2,0.1"];
    edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

    subgraph cluster_hardware {
        label="Host Infrastructure Tier";
        fontcolor="#38BDF8";
        style=dashed;
        color="#334155";
        hw [label="Physical Hardware (CPU / Memory / Disk / NIC)"];
        host_os [label="Host Operating System (Linux Kernel)"];
    }

    subgraph cluster_docker {
        label="Container Execution Core (OS Virtualization)";
        fontcolor="#F59E0B";
        style=dashed;
        color="#334155";
        docker_engine [label="Docker Engine (containerd / runc)\\nShared Host Kernel"];
        c1 [label="Container A (App + Bin/Lib)\\nNamespaces + Cgroups", fillcolor="#064E3B"];
        c2 [label="Container B (App + Bin/Lib)\\nNamespaces + Cgroups", fillcolor="#064E3B"];
    }

    hw -> host_os [label="Powers"];
    host_os -> docker_engine [label="Kernel Primitives"];
    docker_engine -> c1 [label="Isolates"];
    docker_engine -> c2 [label="Isolates"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

# --- SECTIONS 2 THROUGH K+1: ROADMAP TOPICS ---
# (Example for Topic 1: Definition of Containerization)
st.header("2. Definition of Containerization", divider="rainbow")
st.markdown("""
**Containerization** is a form of operating system-level virtualization where multiple isolated user-space
instances (containers) run on a single host, sharing the host operating system's kernel. Unlike Virtual Machines
which emulate virtual hardware and boot an entire guest OS (consuming gigabytes of RAM and taking minutes to start),
containers encapsulate only the application code, runtime libraries, and environment variables.
""")

with st.expander("🔍 View Flow Diagram: Kernel Sharing vs Guest OS", expanded=False):
    dot_sub = """
    digraph IsolationFlow {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10, fontcolor="#F8FAFC", fillcolor="#1E293B", color="#334155"];
        edge [color="#38BDF8"];
        kernel [label="Shared Host Linux Kernel"];
        ns [label="Linux Namespaces\\n(PID, NET, MNT, IPC, UTS)"];
        cg [label="Control Groups (cgroups)\\n(CPU & Memory Limits)"];
        proc [label="Container Process"];
        kernel -> ns -> proc;
        kernel -> cg -> proc;
    }
    """
    st.graphviz_chart(dot_sub, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current Industry Stacks Implement This
In modern cloud architectures, containers are packaged strictly according to Open Container Initiative (OCI) specifications
and executed by lightweight runtimes (such as containerd and CRI-O). Modern infrastructure rejects 2GB monolithic images in
favor of multi-stage Distroless builds (<50MB) with pinned cryptographic checksums, automated vulnerability scanning in
CI/CD, and rootless process enforcement.
""")

df_compare = pd.DataFrame({
    "Dimension": ["Virtual Machines (Type 2)", "Containers (Docker / OCI)"],
    "Architecture": ["Emulated hardware with full Guest OS", "Shared Host OS kernel with isolated namespaces"],
    "Startup Latency": ["Minutes (boots full OS kernel)", "Milliseconds (spawns single process tree)"],
    "Storage Footprint": ["10GB - 50GB per instance", "10MB - 500MB per image layer stack"],
    "Resource Overhead": ["High (dedicated RAM and CPU slices)", "Near-zero (shares kernel memory dynamically)"]
})
st.table(df_compare)

# --- SECTION K+2: PRODUCTION IMPLEMENTATION ---
st.header("3. Production Implementation: Dockerfile & Compose Verification", divider="rainbow")
st.code("""
# Dockerfile (Multi-stage Python FastAPI Service)
# Stage 1: Build & dependency compilation
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Minimal runtime image
FROM python:3.11-slim AS runtime
WORKDIR /app
# Run as non-root user for security hardening
RUN useradd -u 8888 -m appuser
COPY --from=builder /install /usr/local
COPY --chown=appuser:appuser src/ .
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
""", language="dockerfile")

# --- SECTION K+3: INTERACTIVE WORKBENCH ---
st.header("4. Interactive Workbench: Dockerfile Layer & Cache Analyzer", divider="rainbow")

class DockerfileAnalyzer:
    def analyze(self, base_img: str, multi_stage: bool, copy_first: str) -> Dict[str, Any]:
        size_mb = 180 if "slim" in base_img else (35 if "alpine" in base_img else 950)
        if multi_stage:
            size_mb = int(size_mb * 0.4)

        cache_optimal = copy_first == "requirements.txt first (Optimal)"
        return {
            "Estimated Image Size (MB)": size_mb,
            "Build Cache Efficiency": "HIGH (Cached deps)" if cache_optimal else "POOR (Invalidated on code change)",
            "Security Posture": "HARDENED (Non-root)" if multi_stage else "STANDARD",
            "Production Ready": "YES" if cache_optimal and multi_stage else "NEEDS OPTIMIZATION"
        }

analyzer = DockerfileAnalyzer()
st.subheader("Configure Container Build Parameters")
col1, col2 = st.columns(2)
with col1:
    base_choice = st.selectbox("Base Image", ["python:3.11", "python:3.11-slim", "python:3.11-alpine"])
    multi_stage_flag = st.checkbox("Enable Multi-Stage Build Pattern", value=True)
with col2:
    copy_strategy = st.radio("COPY Instruction Ordering", [
        "requirements.txt first (Optimal)",
        "COPY . . before pip install (Cache Busting Anti-pattern)"
    ])

if st.button("🚀 Analyze Build Profile", type="primary"):
    res = analyzer.analyze(base_choice, multi_stage_flag, copy_strategy)
    st.write("### Container Build Assessment")
    st.json(res)

# --- SECTION K+4: TECHNICAL INTERVIEW PREPARATION ---
st.header("5. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qa = [
    ("How does a container achieve process isolation from the host OS?",
     "Containers rely on Linux kernel primitives: Namespaces (PID, NET, MNT, IPC, UTS, USER) partition system resources so a process sees only its own environment, while Control Groups (cgroups) allocate and limit CPU, memory, and I/O resources."),
    ("What is the difference between CMD and ENTRYPOINT in a Dockerfile?",
     "ENTRYPOINT defines the immutable core executable that runs when the container starts. CMD provides default arguments that can be easily overridden by arguments passed to 'docker run'. When used together, ENTRYPOINT is the command and CMD provides the default flags."),
    ("Why does Dockerfile instruction ordering affect build times dramatically?",
     "Docker executes builds layer-by-layer and caches results. If an early instruction (e.g. COPY . .) changes, all subsequent cached layers are invalidated. Placing slowly-changing instructions (COPY requirements.txt; RUN pip install) before source code ensures fast builds using cached dependency layers."),
    ("What is the difference between a Named Volume and a Bind Mount?",
     "Named Volumes are managed directly by Docker within /var/lib/docker/volumes/, ensuring portability, proper permission handling, and driver support. Bind Mounts map a specific host directory directly into the container, exposing host-dependent file paths ideal for local live-reload development."),
    ("How do containers communicate in Docker Compose without knowing each other's IP addresses?",
     "Docker Compose automatically provisions an isolated custom bridge network. The embedded Docker DNS server resolves service names (e.g. 'db' or 'redis') defined in the Compose file directly to their respective container IP addresses.")
]

for i, (q, a) in enumerate(interview_qa, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Breaking Error)**: Exposing Ports Without Publishing. Writing EXPOSE 8000 in a Dockerfile only documents the port; it does not map it to the host. You must pass '-p 8000:8000' at runtime.")
    st.error("**Trap 2 (Breaking Error)**: Database Data Loss on Container Removal. Running a database container without a named volume stores data in the ephemeral writable layer, which is permanently deleted when the container is removed.")
with col_trap2:
    st.warning("**Trap 3 (Performance Warning)**: Bloated Image from Missing .dockerignore. Forgetting a .dockerignore file causes large local artifacts (.git, venv, model checkpoints) to be sent into the build context, slowing down builds.")
    st.warning("**Trap 4 (Security Warning)**: Running Containers as Root. Defaulting to the root user inside containers increases the blast radius if an application has a remote code execution vulnerability. Always specify USER.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("🎴 Flashcard 1: Linux Namespaces"):
        st.markdown("**PID (processes), NET (networking), MNT (filesystems), IPC (inter-process), UTS (hostname).**")
    with st.expander("🎴 Flashcard 2: Overlay2 UnionFS"):
        st.markdown("**Stacks read-only image layers under a thin, ephemeral writable container layer (Copy-on-Write).**")
with fc_col2:
    with st.expander("🎴 Flashcard 3: Docker Compose DNS"):
        st.markdown("**Embedded DNS resolves service keys directly to container IPs on user-defined bridge networks.**")
    with st.expander("🎴 Flashcard 4: Multi-Stage Build"):
        st.markdown("**Separates heavyweight compile-time SDKs from the minimal runtime image using COPY --from=builder.**")

st.divider()

# --- SECTION K+5: SELF-ASSESSMENT CHECKLIST ---
st.header("6. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can articulate the architectural difference between a Container and a VM.", value=True)
chk2 = st.checkbox("I understand Linux namespaces and control groups (cgroups).", value=True)
chk3 = st.checkbox("I know how to optimize Docker build caching by ordering Dockerfile instructions.", value=True)
chk4 = st.checkbox("I understand the difference between Named Volumes and Bind Mounts.", value=True)
chk5 = st.checkbox("I can write a production-ready, multi-stage Dockerfile with non-root security.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on this lesson. Ready for the next module!")
```
