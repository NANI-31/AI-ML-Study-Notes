import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Architecture Diagrams Showcase",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ Architecture Diagrams Showcase — Top 4 Visualization Libraries")
st.markdown(
    """
This module demonstrates the **Top 4 libraries** for rendering professional, interactive,
and publication-grade system architecture diagrams directly inside Streamlit applications.
"""
)

st.divider()

# ==============================================================================
# Helper for Mermaid.js Rendering
# ==============================================================================
def render_mermaid(code: str, height: int = 450):
    """Render Mermaid.js diagrams using CDN without external pip dependencies."""
    html_code = f"""
    <div style="background-color: transparent; text-align: center;">
        <div class="mermaid">
{code}
        </div>
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'dark',
            securityLevel: 'loose',
            themeVariables: {{
                fontFamily: 'Helvetica, Arial, sans-serif',
                primaryColor: '#3B82F6',
                primaryTextColor: '#FFFFFF',
                primaryBorderColor: '#60A5FA',
                lineColor: '#38BDF8',
                secondaryColor: '#10B981',
                tertiaryColor: '#1E293B'
            }}
        }});
    </script>
    """
    components.html(html_code, height=height)

# ==============================================================================
# 1. st.graphviz_chart (Native Streamlit Built-in)
# ==============================================================================
st.header("1. Native Graphviz (`st.graphviz_chart`)", divider="rainbow")
st.markdown(
    """
- **Installation**: **Zero extra installs** (Built into Streamlit core).
- **Technology**: Uses WebAssembly (Viz.js) inside the browser; requires no system Graphviz binaries.
- **Output**: Crisp, zoomable vector SVG diagrams using the standard DOT language.
"""
)

col_gv_ctrl1, col_gv_ctrl2 = st.columns(2)
with col_gv_ctrl1:
    gv_direction = st.radio(
        "Layout Flow Direction:",
        options=["TB (Top-to-Bottom)", "LR (Left-to-Right)"],
        horizontal=True,
        key="gv_dir_radio",
    )
    direction_code = "TB" if "TB" in gv_direction else "LR"

with col_gv_ctrl2:
    gv_theme = st.selectbox(
        "Node Color Palette:",
        options=["Modern Slate (Dark)", "Emerald Green", "Indigo Blue"],
        key="gv_theme_select",
    )

fill_primary = "#1E293B" if "Slate" in gv_theme else ("#065F46" if "Emerald" in gv_theme else "#312E81")
border_primary = "#475569" if "Slate" in gv_theme else ("#10B981" if "Emerald" in gv_theme else "#6366F1")

dot_code = f"""
digraph SystemArchitecture {{
    rankdir={direction_code};
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fillcolor="{fill_primary}", fontcolor="#F8FAFC", fontname="Helvetica", penwidth=1.5, color="{border_primary}"];
    edge [color="#38BDF8", fontcolor="#94A3B8", fontname="Helvetica", fontsize=10];

    subgraph cluster_clients {{
        label="Client Tier";
        fontcolor="#94A3B8";
        style="dashed";
        color="#475569";
        WebClient [label="Web Browser\\n(Streamlit SPA)", fillcolor="#0E7490"];
        MobileApp [label="Mobile Client\\n(React Native)", fillcolor="#0E7490"];
    }}

    subgraph cluster_gateway {{
        label="Ingress & Routing";
        fontcolor="#94A3B8";
        style="rounded";
        color="#475569";
        Nginx [label="Reverse Proxy & LB\\n(Nginx / Envoy)", shape=component, fillcolor="#1E3A8A"];
    }}

    subgraph cluster_backend {{
        label="Service Mesh";
        fontcolor="#94A3B8";
        style="rounded";
        color="#475569";
        AppServer [label="Streamlit Core Server\\n(Tornado Async)", fillcolor="#4338CA"];
        MLService [label="ML Inference Engine\\n(FastAPI / Triton)", fillcolor="#7C3AED"];
    }}

    subgraph cluster_storage {{
        label="Data & Persistence";
        fontcolor="#94A3B8";
        style="rounded";
        color="#475569";
        RedisCache [label="Redis Feature Store\\n(In-Memory Cache)", shape=cylinder, fillcolor="#991B1B"];
        PostgresDB [label="PostgreSQL DB\\n(Metadata & Logs)", shape=cylinder, fillcolor="#15803D"];
    }}

    WebClient -> Nginx [label="WebSocket / HTTPS"];
    MobileApp -> Nginx [label="REST API"];
    Nginx -> AppServer [label="WebSocket Forward"];
    Nginx -> MLService [label="gRPC / HTTP"];
    AppServer -> RedisCache [label="Session Lookup"];
    MLService -> RedisCache [label="Get Embeddings"];
    MLService -> PostgresDB [label="Audit Prediction"];
}}
"""

col_gv1, col_gv2 = st.columns([3, 2])
with col_gv1:
    st.markdown("#### 🖥️ Rendered Vector Diagram")
    st.graphviz_chart(dot_code)

with col_gv2:
    st.markdown("#### 💻 Source DOT Code")
    st.code(dot_code, language="dot")

st.divider()

# ==============================================================================
# 2. Mermaid.js (Sequence & State Diagrams via CDN)
# ==============================================================================
st.header("2. Mermaid.js (Interactive Sequence & Flow Diagrams)", divider="rainbow")
st.markdown(
    """
- **Installation**: **Zero extra pip packages** (Rendered via clean HTML/JavaScript CDN).
- **Best For**: **Sequence diagrams**, execution lifecycles, and state machines.
"""
)

mermaid_tabs = st.tabs(["🔄 Sequence Diagram (WebSocket Lifecycle)", "📊 State Machine (ML Model Lifecycle)"])

with mermaid_tabs[0]:
    seq_code = """sequenceDiagram
    autonumber
    actor Browser as Web Browser (React)
    participant Tornado as Tornado Web Server
    participant Worker as ScriptRunner Worker
    participant Storage as In-Memory State

    Browser->>Tornado: HTTP GET /_stcore/stream
    Tornado-->>Browser: 101 Switching Protocols (WebSocket Open)
    Browser->>Tornado: BackMsg: Widget Interaction (slider=42)
    Tornado->>Worker: Spawn / Resume ScriptRunner Thread
    Worker->>Storage: Read st.session_state
    Worker->>Worker: Execute app.py Top-to-Bottom
    Worker-->>Tornado: Emit ForwardMsg (UI Deltas)
    Tornado-->>Browser: WebSocket Push Protobuf Deltas
    Browser->>Browser: React Updates Virtual DOM"""

    col_seq1, col_seq2 = st.columns([3, 2])
    with col_seq1:
        st.markdown("#### 🖥️ Rendered Sequence Diagram")
        render_mermaid(seq_code, height=480)
    with col_seq2:
        st.markdown("#### 💻 Mermaid Syntax")
        st.code(seq_code, language="mermaid")

with mermaid_tabs[1]:
    state_code = """stateDiagram-v2
    [*] --> Ingestion: Raw Dataset Upload
    Ingestion --> FeatureEngineering: Preprocessing and Scaling
    FeatureEngineering --> Training: Fit Estimators
    Training --> Validation: Cross-Validation
    Validation --> ModelRegistry: Pass (F1 score >= 0.90)
    Validation --> Training: Fail (F1 score below 0.90 - Retune)
    ModelRegistry --> Staging: Canary Deployment
    Staging --> Production: Champion Promotion
    Production --> [*]: Deprecated or Retrained"""

    col_st1, col_st2 = st.columns([3, 2])
    with col_st1:
        st.markdown("#### 🖥️ Rendered State Machine")
        render_mermaid(state_code, height=420)
    with col_st2:
        st.markdown("#### 💻 Mermaid Syntax")
        st.code(state_code, language="mermaid")

st.divider()

# ==============================================================================
# 3. Pyvis (Physics-Based Interactive Force Graphs)
# ==============================================================================
st.header("3. Pyvis (Physics-Based Interactive Network Graphs)", divider="rainbow")
st.markdown(
    """
- **Installation**: `pip install pyvis`
- **Interactivity**: **Live spring-physics simulation**. You can **drag, bounce, and rearrange** nodes with your mouse!
- **Best For**: Microservice dependency graphs, knowledge graphs, and neural topology visualization.
"""
)

# Build Interactive Pyvis Graph in memory
net = Network(height="450px", width="100%", bgcolor="#0F172A", font_color="#F8FAFC")
net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=120, spring_strength=0.04, damping=0.09)

# Add Nodes with Custom Colors and Shapes
net.add_node(1, label="Browser Client\n(React)", shape="box", color="#06B6D4", size=25)
net.add_node(2, label="Streamlit Hub\n(Port 8501)", shape="box", color="#3B82F6", size=30)
net.add_node(3, label="Tornado Server\n(Async Loop)", shape="ellipse", color="#6366F1", size=22)
net.add_node(4, label="Worker Thread 1\n(Session A)", shape="dot", color="#A855F7", size=18)
net.add_node(5, label="Worker Thread 2\n(Session B)", shape="dot", color="#A855F7", size=18)
net.add_node(6, label="Redis Cache\n(Feature Store)", shape="database", color="#EF4444", size=22)
net.add_node(7, label="PostgreSQL\n(Metadata DB)", shape="database", color="#10B981", size=22)
net.add_node(8, label="LLM Endpoint\n(Ollama / OpenAI)", shape="star", color="#F59E0B", size=25)

# Add Edges
net.add_edge(1, 2, label="WebSocket", color="#38BDF8")
net.add_edge(2, 3, label="Dispatches", color="#818CF8")
net.add_edge(3, 4, label="Spawns", color="#C084FC")
net.add_edge(3, 5, label="Spawns", color="#C084FC")
net.add_edge(4, 6, label="Read Cache", color="#F87171")
net.add_edge(4, 8, label="Chat Prompt", color="#FBBF24")
net.add_edge(5, 7, label="Query Log", color="#34D399")

pyvis_html = net.generate_html()

col_pv1, col_pv2 = st.columns([3, 2])
with col_pv1:
    st.markdown("#### 🖥️ Interactive Draggable Canvas *(Click & Drag Nodes!)*")
    components.html(pyvis_html, height=470)

with col_pv2:
    st.markdown("#### 💻 Pyvis Source Code")
    pyvis_snippet = """from pyvis.network import Network
import streamlit.components.v1 as components

net = Network(height='450px', width='100%', bgcolor='#0F172A', font_color='white')
net.barnes_hut(gravity=-3000)

net.add_node(1, label='Browser', shape='box', color='#06B6D4')
net.add_node(2, label='Streamlit Server', shape='box', color='#3B82F6')
net.add_node(3, label='Redis Cache', shape='database', color='#EF4444')
net.add_edge(1, 2, label='WebSocket')
net.add_edge(2, 3, label='Cache Query')

# Render in Streamlit without writing to disk
components.html(net.generate_html(), height=470)"""
    st.code(pyvis_snippet, language="python")

st.divider()

# ==============================================================================
# 4. Diagrams by Mingrammer (Cloud Architecture as Code)
# ==============================================================================
st.header("4. `diagrams` by Mingrammer (Cloud Architecture as Code)", divider="rainbow")
st.markdown(
    """
- **Installation**: `pip install diagrams`
- **Why It’s Famous**: Write pure Python code to produce system architecture diagrams decorated with **official cloud provider icons** (AWS, GCP, Azure, Kubernetes, On-Premise, SaaS).
- **Requirement**: Requires the Graphviz system binary (`dot.exe`) installed on the host machine PATH.
"""
)

col_diag1, col_diag2 = st.columns(2)

with col_diag1:
    st.markdown("#### 💻 Production Cloud Architecture Code")
    diagrams_code = """from diagrams import Diagram, Cluster
from diagrams.programming.framework import React
from diagrams.programming.language import Python
from diagrams.onprem.network import Nginx
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.inmemory import Redis

with Diagram("Production Streamlit ML Architecture", show=False, filename="streamlit_cloud_arch"):
    with Cluster("Client Layer"):
        client = React("Web SPA")

    with Cluster("DMZ / Ingress"):
        proxy = Nginx("Load Balancer")

    with Cluster("Compute Layer"):
        app = Python("Streamlit Server (Tornado)")
        workers = [
            Python("Worker Session 1"),
            Python("Worker Session 2")
        ]

    with Cluster("Persistence Layer"):
        cache = Redis("Session Cache")
        db = PostgreSQL("App Database")

    client >> proxy >> app
    app >> workers
    workers >> cache
    workers >> db"""
    st.code(diagrams_code, language="python")

with col_diag2:
    st.markdown("#### 📐 Architecture Topology Representation")
    # Clean visual presentation of the cloud architecture topology
    cloud_dot = """
    digraph CloudTopology {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style="rounded,filled", fillcolor="#1E293B", fontcolor="white", fontname="Helvetica", color="#475569"];
        edge [color="#60A5FA", fontcolor="#94A3B8", fontname="Helvetica", fontsize=9];

        subgraph cluster_client {
            label="[React] Client Layer";
            color="#38BDF8";
            style="dashed";
            Client [label="🖥️ Web Browser\\n(React Client)", fillcolor="#0E7490"];
        }

        subgraph cluster_ingress {
            label="[Nginx] Ingress Layer";
            color="#10B981";
            style="dashed";
            Ingress [label="🛡️ Nginx Proxy\\n(SSL / LB)", fillcolor="#065F46"];
        }

        subgraph cluster_app {
            label="[Python] Application Cluster";
            color="#818CF8";
            style="dashed";
            App [label="⚡ Streamlit App\\n(Tornado Async)", fillcolor="#312E81"];
            Worker [label="⚙️ ScriptRunner\\nWorker Threads", fillcolor="#4338CA"];
        }

        subgraph cluster_db {
            label="[Data] Storage Tier";
            color="#F43F5E";
            style="dashed";
            Redis [label="💾 Redis Cache\\n(Feature Store)", shape=cylinder, fillcolor="#9F1239"];
            SQL [label="🗄️ PostgreSQL\\n(Logs & State)", shape=cylinder, fillcolor="#166534"];
        }

        Client -> Ingress -> App -> Worker;
        Worker -> Redis [label="Cache"];
        Worker -> SQL [label="Query"];
    }
    """
    st.graphviz_chart(cloud_dot)
    st.caption("Rendered diagram matching the exact node topology of the `diagrams` code on the left.")

st.divider()

# ==============================================================================
# 5. Master Architecture Library Comparison
# ==============================================================================
st.header("📊 Master Comparison: Which Diagramming Library Should You Use?", divider="rainbow")

comp_data = {
    "Library": [
        "st.graphviz_chart",
        "Mermaid.js",
        "Pyvis",
        "diagrams (Mingrammer)"
    ],
    "Installation": [
        "Zero extra installs (Built into Streamlit)",
        "Zero extra installs (via components.html)",
        "pip install pyvis",
        "pip install diagrams + system Graphviz"
    ],
    "Interactivity Level": [
        "Zoomable & Pan-capable SVG",
        "Static crisp SVG layout",
        "Physics-based draggable & bouncy nodes",
        "Static high-res PNG / SVG output"
    ],
    "Primary Superpower": [
        "Fastest setup; native DOT language syntax",
        "Best for sequence flows & state machines",
        "Interactive topology & relationship graphs",
        "Official cloud provider icons (AWS, GCP, K8s)"
    ],
    "Best Project Use Case": [
        "Internal algorithm flowcharts & component trees",
        "API handshake & request-response sequence lifecycles",
        "Microservices, network topologies & graph databases",
        "Production cloud architecture & DevOps presentations"
    ]
}

st.dataframe(comp_data, width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# 6. Direct Head-to-Head Comparison: The Same Diagram in Graphviz vs. Pyvis
# ==============================================================================
st.header("6. ⚔️ Direct Head-to-Head: The Exact Same Diagram in Graphviz vs. Pyvis", divider="rainbow")
st.markdown(
    """
To directly compare how both visualization engines handle the exact same architecture, 
the identical **7-tier ML Microservices System** is rendered below using both libraries:
- **Left Column**: Rendered with **Graphviz (`st.graphviz_chart`)** — Deterministic, hierarchical vector layout.
- **Right Column**: Rendered with **Pyvis (`Network` via `components.html`)** — Interactive force-directed physics layout with draggable nodes.
"""
)

col_cmp_ctrl1, col_cmp_ctrl2 = st.columns(2)
with col_cmp_ctrl1:
    h2h_direction = st.radio(
        "Graphviz Flow Direction:",
        options=["LR (Left-to-Right)", "TB (Top-to-Bottom)"],
        horizontal=True,
        key="h2h_gv_dir",
    )
    h2h_dir_code = "LR" if "LR" in h2h_direction else "TB"

with col_cmp_ctrl2:
    physics_enabled = st.checkbox("Enable Pyvis Physics Simulation", value=True, key="h2h_physics")

# --- 1. Graphviz DOT Definition ---
h2h_dot = f"""
digraph SystemComparison {{
    rankdir={h2h_dir_code};
    bgcolor="transparent";
    node [fontname="Helvetica", fontsize=11, penwidth=1.5, margin="0.2,0.1"];
    edge [color="#38BDF8", fontcolor="#94A3B8", fontname="Helvetica", fontsize=9, arrowsize=0.8];

    WebClient [label="🌐 Web Browser\\n(Client UI)", shape=box, style="rounded,filled", fillcolor="#0E7490", fontcolor="#FFFFFF", color="#22D3EE"];
    Gateway [label="🛡️ API Gateway\\n(Nginx / Envoy)", shape=hexagon, style="filled", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
    StreamlitApp [label="⚡ Streamlit App\\n(Frontend Svc)", shape=box, style="rounded,filled", fillcolor="#4338CA", fontcolor="#FFFFFF", color="#818CF8"];
    InferenceEngine [label="🧠 FastAPI Engine\\n(ML Inference)", shape=box, style="rounded,filled", fillcolor="#7C3AED", fontcolor="#FFFFFF", color="#A78BFA"];
    RedisCache [label="💾 Redis Cache\\n(Feature Store)", shape=cylinder, style="filled", fillcolor="#991B1B", fontcolor="#FFFFFF", color="#F87171"];
    PostgresDB [label="🗄️ PostgreSQL\\n(Inference Logs)", shape=cylinder, style="filled", fillcolor="#15803D", fontcolor="#FFFFFF", color="#4ADE80"];
    ModelStore [label="📦 S3 Bucket\\n(Model Weights)", shape=box, style="rounded,filled", fillcolor="#D97706", fontcolor="#FFFFFF", color="#FBBF24"];

    WebClient -> Gateway [label="HTTPS/WS"];
    Gateway -> StreamlitApp [label="Proxy :8501"];
    Gateway -> InferenceEngine [label="REST /predict"];
    StreamlitApp -> InferenceEngine [label="gRPC / Call"];
    InferenceEngine -> RedisCache [label="Get Features"];
    InferenceEngine -> ModelStore [label="Load ONNX"];
    InferenceEngine -> PostgresDB [label="Audit Logs"];
}}
"""

# --- 2. Pyvis Network Definition (Exact Same 7 Nodes & Edges) ---
net_h2h = Network(height="480px", width="100%", bgcolor="#0F172A", font_color="#F8FAFC")
if physics_enabled:
    net_h2h.barnes_hut(gravity=-2800, central_gravity=0.3, spring_length=120, spring_strength=0.04, damping=0.09)
else:
    net_h2h.toggle_physics(False)

nodes_config = [
    (1, "🌐 Web Browser\n(Client UI)", "box", "#0E7490", "#22D3EE"),
    (2, "🛡️ API Gateway\n(Nginx / Envoy)", "hexagon", "#1E3A8A", "#60A5FA"),
    (3, "⚡ Streamlit App\n(Frontend Svc)", "box", "#4338CA", "#818CF8"),
    (4, "🧠 FastAPI Engine\n(ML Inference)", "box", "#7C3AED", "#A78BFA"),
    (5, "💾 Redis Cache\n(Feature Store)", "database", "#991B1B", "#F87171"),
    (6, "🗄️ PostgreSQL\n(Inference Logs)", "database", "#15803D", "#4ADE80"),
    (7, "📦 S3 Bucket\n(Model Weights)", "box", "#D97706", "#FBBF24"),
]

for node_id, label, shape, color, border in nodes_config:
    net_h2h.add_node(
        node_id,
        label=label,
        shape=shape,
        color={"background": color, "border": border, "highlight": {"background": "#F59E0B", "border": "#FFFFFF"}},
        font={"color": "#FFFFFF", "face": "Helvetica", "size": 13},
        borderWidth=2,
        size=26,
    )

edges_config = [
    (1, 2, "HTTPS/WS"),
    (2, 3, "Proxy :8501"),
    (2, 4, "REST /predict"),
    (3, 4, "gRPC / Call"),
    (4, 5, "Get Features"),
    (4, 7, "Load ONNX"),
    (4, 6, "Audit Logs"),
]

for src, dst, lbl in edges_config:
    net_h2h.add_edge(
        src,
        dst,
        label=lbl,
        color="#38BDF8",
        arrows="to",
        font={"color": "#94A3B8", "size": 11, "align": "horizontal"},
    )

pyvis_h2h_html = net_h2h.generate_html()

# --- 3. Side-by-Side Display ---
col_cmp_gv, col_cmp_pv = st.columns(2)

with col_cmp_gv:
    st.markdown("#### 1️⃣ Graphviz (`st.graphviz_chart`)")
    st.graphviz_chart(h2h_dot)
    st.caption("✅ Deterministic layout, sharp vector scaling, zero dependencies, hierarchical tier alignment.")

with col_cmp_pv:
    st.markdown("#### 2️⃣ Pyvis (`components.html`)")
    components.html(pyvis_h2h_html, height=480)
    st.caption("✅ Draggable physics canvas, zoom/pan controls, dynamic force-directed simulation (try dragging nodes!).")

# --- 4. Architectural Comparison Scorecard ---
st.markdown("### 📋 Graphviz vs. Pyvis Feature Breakdown")
scorecard = {
    "Dimension": [
        "Rendering Technology",
        "Layout Nature",
        "Interactivity",
        "Dependency Footprint",
        "Subgraphs / Tier Clusters",
        "Export / Save Quality",
        "Ideal Architectural Use Case",
    ],
    "Graphviz (st.graphviz_chart)": [
        "Vector SVG via WebAssembly (Viz.js in browser)",
        "Strict hierarchical DAG (Directed Acyclic Graph)",
        "Static (browser zoom and pan only)",
        "Built-in to Streamlit core (0 extra pip packages)",
        "Native support via 'subgraph cluster_name'",
        "Lossless scalable vector graphic (SVG)",
        "Rigid enterprise pipelines, data workflows & UML diagrams",
    ],
    "Pyvis (vis.js Canvas in iframe)": [
        "HTML5 Canvas with Barnes-Hut physics simulation",
        "Dynamic force-directed network graph",
        "Fully interactive (click, drag, bounce, zoom)",
        "Requires 'pip install pyvis'",
        "Physics clustering (no visual container boxes)",
        "Canvas raster rendering",
        "Network topologies, knowledge graphs & microservice meshes",
    ],
}
st.dataframe(scorecard, width="stretch", hide_index=True)

