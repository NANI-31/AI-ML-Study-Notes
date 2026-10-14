import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="1.1 What is Streamlit? | Complete Roadmap Guide",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ 1.1 What is Streamlit? — Core Concepts, Comparisons & Real-World Use Cases")
st.markdown(
    """
Welcome to **Lesson 1.1: What is Streamlit?**
This module provides a comprehensive technical breakdown of the Streamlit framework,
its architectural philosophy, deep comparisons against other web frameworks, core advantages,
and real-world industry applications.
"""
)

st.divider()

# ==============================================================================
# 1. Definition of Streamlit
# ==============================================================================
st.header("1. Definition of Streamlit", divider="rainbow")

col_def1, col_def2 = st.columns([3, 2])

with col_def1:
    st.markdown(
        """
### What is Streamlit?
**Streamlit** is an open-source Python framework designed specifically for **Data Scientists, Machine Learning Engineers, and GenAI Developers** to build interactive, beautiful, and data-driven web applications in pure Python—with **zero frontend experience** required.

#### Core Architectural Philosophy:
1. **Embrace Python Scripting**: You write standard procedural Python code. Every line executes sequentially from top to bottom.
2. **Treat Widgets as Variables**: An interactive UI element (like a slider or dropdown) is declared like a normal Python variable assignment (`x = st.slider(...)`).
3. **Reactive Top-to-Bottom Execution**: Whenever a user interacts with a widget, Streamlit automatically reruns the entire Python script from top to bottom, updating only the elements that changed via WebSockets.
4. **Effortless State & Caching**: Stateful workflows are managed via `st.session_state`, while expensive computations and model weights are cached using `@st.cache_data` and `@st.cache_resource`.
"""
    )

with col_def2:
    st.markdown("#### 🧪 Interactive Hello World Demonstration")
    st.code(
        '''import streamlit as st

st.subheader("Interactive Square Calculator")
val = st.slider("Select a number", min_value=1, max_value=50, value=7)
st.write(f"The square of **{val}** is **{val**2}**.")''',
        language="python",
    )
    # Live Interactive Execution
    st.markdown("##### 🖥️ Live Output:")
    calc_val = st.slider("Select a number", min_value=1, max_value=50, value=7, key="demo_calc_slider")
    st.success(f"The square of **{calc_val}** is **{calc_val**2}**.")

st.divider()

# ==============================================================================
# 2. Why Streamlit is Important
# ==============================================================================
st.header("2. Why Streamlit is Important", divider="rainbow")

st.markdown(
    """
In traditional software and data science workflows, sharing a model or insight was notoriously slow and fragmented.
Streamlit revolutionized the ecosystem by eliminating the **Frontend-Backend Bottleneck**.
"""
)

col_why1, col_why2, col_why3 = st.columns(3)

with col_why1:
    st.markdown(
        """
#### 🚫 The Old Painful Way
- **Jupyter Notebooks**: Hard to share with non-technical stakeholders; code and outputs are cluttered.
- **Frontend Dependency**: Building a UI required full-stack developers (React/Vue + HTML/CSS).
- **Backend API Glue**: Required building Flask/FastAPI REST endpoints and custom serialization.
- **Slow Iteration**: Prototypes took weeks or months to reach stakeholders.
"""
    )

with col_why2:
    st.markdown(
        """
#### ⚡ The Streamlit Revolution
- **100% Pure Python**: The data scientist builds the entire interface and logic in a single language.
- **Zero Frontend Code**: No HTML tags, CSS styling, or JavaScript event listeners needed.
- **Minutes to Prototype**: Turn an exploratory script or ML pipeline into a sharable app in 30 minutes.
- **Stakeholder Accessibility**: Clean, polished web interface accessible to anyone via a web browser.
"""
    )

with col_why3:
    st.markdown(
        """
#### 💼 Strategic Business Impact
- **Immediate Feedback Loops**: Stakeholders test assumptions and provide feedback on day one.
- **Cost Reduction**: Eliminates the overhead of assigning dedicated web developers for internal tools.
- **Snowflake Native**: Fully integrated into Snowflake's Data Cloud (*Streamlit in Snowflake*) for secure enterprise deployment.
- **GenAI Accelerator**: Native chat primitives enable instant prototyping of LLM chatbots and RAG agents.
"""
    )

st.divider()

# ==============================================================================
# 3-6. Streamlit vs Alternative Web Frameworks
# ==============================================================================
st.header("3–6. Streamlit vs Alternative Web Frameworks", divider="rainbow")
st.markdown(
    "Understanding when to choose Streamlit versus Flask, Django, Dash, or Gradio is a fundamental technical interview topic."
)

tab_matrix, tab_flask, tab_django, tab_dash, tab_gradio = st.tabs(
    [
        "📊 Master Framework Matrix",
        "⚖️ Streamlit vs Flask",
        "⚖️ Streamlit vs Django",
        "⚖️ Streamlit vs Dash",
        "⚖️ Streamlit vs Gradio",
    ]
)

with tab_matrix:
    st.markdown("### Master Framework Comparison Matrix")
    matrix_df = pd.DataFrame(
        {
            "Feature / Dimension": [
                "Primary Audience",
                "UI Building Language",
                "Execution Model",
                "Routing & URLs",
                "State Management",
                "Database / ORM",
                "Speed of Prototyping",
                "Best Used For",
            ],
            "Streamlit": [
                "Data Scientists & ML/AI Engineers",
                "Pure Python (Declarative)",
                "Top-to-bottom reactive reruns",
                "Built-in Multi-page apps (`pages/`)",
                "Native `st.session_state`",
                "None (Integrates with any DB via Python)",
                "Ultra Fast (Minutes)",
                "Data dashboards, ML demos, GenAI chat UIs",
            ],
            "Flask": [
                "Backend & Full-Stack Developers",
                "HTML + CSS + Jinja2 Templates",
                "Request/Response HTTP cycle",
                "Explicit route decorators (`@app.route`)",
                "Sessions, cookies, Redis tokens",
                "None built-in (SQLAlchemy optional)",
                "Moderate (Requires manual HTML/JS)",
                "Custom REST APIs, microservices, lightweight web apps",
            ],
            "Django": [
                "Full-Stack Web Developers",
                "HTML Templates or REST APIs (DRF)",
                "Standard MVC / MVT architecture",
                "Explicit URL configuration (`urls.py`)",
                "Built-in session engine & auth",
                "Built-in Django ORM + auto-migrations",
                "Slower (High setup boilerplate)",
                "Large enterprise web apps, SaaS portals, e-commerce",
            ],
            "Dash (Plotly)": [
                "Data Analysts & BI Engineers",
                "Python wrapped around React/Plotly",
                "Explicit reactive callbacks (`@app.callback`)",
                "Custom multi-page plugin setup",
                "`dcc.Store` and browser storage",
                "None built-in",
                "Moderate (Significant callback boilerplate)",
                "Complex financial dashboards, enterprise BI analytics",
            ],
            "Gradio": [
                "ML Researchers & HuggingFace Users",
                "Pure Python (`gr.Interface` / `gr.Blocks`)",
                "Function-wrapping event listeners",
                "Single page or tabbed blocks",
                "Component state / session objects",
                "None built-in",
                "Ultra Fast (Minutes)",
                "Quick single-model inference demos, HuggingFace Spaces",
            ],
        }
    )
    st.dataframe(matrix_df, width="stretch", hide_index=True)

with tab_flask:
    st.markdown("### 3. Streamlit vs Flask")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown(
            """
#### ⚡ Streamlit
- **Focus**: Building interactive frontend UIs and dashboards for data and ML models.
- **Workflow**: 100% Python. No HTML templates, CSS files, or JavaScript fetch calls.
- **Architecture**: WebSocket-driven reactive execution. State is held in Python memory.
- **Limitation**: Not designed for serving high-throughput REST APIs, webhooks, or public user accounts.
"""
        )
    with col_f2:
        st.markdown(
            """
#### 🌶️ Flask
- **Focus**: Microframework for building RESTful backend APIs, microservices, and custom web applications.
- **Workflow**: Explicit routing (`@app.route('/predict', methods=['POST'])`), JSON serialization, request validation.
- **Architecture**: Stateless HTTP request-response lifecycle.
- **Limitation**: Has zero built-in UI components. Requires manual frontend implementation (React, Vue, or Jinja2 templates).
"""
        )
    st.info("💡 **Rule of Thumb**: Use **Streamlit** when you need an interactive UI for your model. Use **Flask/FastAPI** when other services or mobile apps need to consume your model via JSON endpoints.")

with tab_django:
    st.markdown("### 4. Streamlit vs Django")
    col_dj1, col_dj2 = st.columns(2)
    with col_dj1:
        st.markdown(
            """
#### ⚡ Streamlit
- **Philosophy**: Lightweight, minimalist, focused solely on the data/ML interface layer.
- **Boilerplate**: Near-zero. A 5-line script is an entire functional application.
- **Database**: No ORM. Reads directly from Pandas, SQL queries, CSVs, or APIs.
- **Authentication**: Lightweight community auth or cloud SSO integrations.
"""
        )
    with col_dj2:
        st.markdown(
            """
#### 🎸 Django
- **Philosophy**: "Batteries-included" monolithic web framework.
- **Boilerplate**: High. Requires project skeleton, `settings.py`, `models.py`, `views.py`, migrations, admin panels.
- **Database**: Powerful built-in Object-Relational Mapper (ORM) with schema migrations.
- **Authentication**: Enterprise-grade built-in user authentication, permissions, and session management.
"""
        )
    st.info("💡 **Rule of Thumb**: Use **Django** for building commercial SaaS products, e-commerce stores, and multi-user transactional websites. Use **Streamlit** for data insights and internal AI tools.")

with tab_dash:
    st.markdown("### 5. Streamlit vs Dash")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(
            """
#### ⚡ Streamlit
- **Syntax**: Declarative and linear. You write code top-to-bottom as if in a script.
- **Interactivity**: Automatic script reruns on widget interaction.
- **Learning Curve**: Minutes. Any Python beginner can build apps immediately.
- **Visuals**: Seamless integration with Matplotlib, Seaborn, Altair, Plotly, PyDeck.
"""
        )
    with col_d2:
        st.markdown(
            """
#### 📊 Dash (Plotly)
- **Syntax**: Explicit reactive programming using `@app.callback(Output(...), Input(...))` decorators.
- **Interactivity**: Fine-grained callback graph. Only specified component callbacks execute.
- **Learning Curve**: Steep. Requires understanding callback inputs/outputs, component IDs, and React-like component structures.
- **Visuals**: Best-in-class for complex, highly customized Plotly interactive charts.
"""
        )
    st.info("💡 **Rule of Thumb**: Use **Streamlit** for 90% of data science and AI applications. Use **Dash** when you require enterprise-grade callback control or deeply customized React UI trees.")

with tab_gradio:
    st.markdown("### 6. Streamlit vs Gradio")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown(
            """
#### ⚡ Streamlit
- **Scope**: Full dashboard and data application development.
- **Layouts**: Highly flexible layout primitives (`st.columns`, `st.sidebar`, `st.expander`, `st.tabs`, `st.container`).
- **Data Display**: Powerful DataFrames, interactive data editing (`st.data_editor`), metric cards, and charts.
- **Ecosystem**: Snowflake, corporate dashboards, BI tools, and multi-page apps.
"""
        )
    with col_g2:
        st.markdown(
            """
#### 🚀 Gradio
- **Scope**: Rapid model benchmarking and model card input/output wrappers (`Input -> Model -> Output`).
- **Layouts**: Primarily centered around `gr.Interface` (Input widgets on the left, Output widgets on the right).
- **Flagging**: Built-in data collection / flagging mechanism for model auditing and reinforcement learning.
- **Ecosystem**: Deep integration with Hugging Face Model Hub and Spaces for sharing deep learning checkpoints.
"""
        )
    st.info("💡 **Rule of Thumb**: Use **Gradio** for quick 5-minute Hugging Face model inference demos. Use **Streamlit** for complete multi-page data applications, analytical dashboards, and business workflows.")

st.divider()

# ==============================================================================
# 7. Advantages of Streamlit
# ==============================================================================
st.header("7. Core Advantages of Streamlit", divider="rainbow")

col_adv1, col_adv2 = st.columns(2)

with col_adv1:
    st.markdown(
        """
#### 1. 🐍 Build Web Apps Using Pure Python
Write 100% standard Python code. Leverage all your favorite Python libraries—such as **NumPy, Pandas, Scikit-Learn, PyTorch, TensorFlow, Hugging Face, LangChain, and LlamaIndex**—directly inside your user interface.

#### 2. 🎨 Zero HTML, CSS, or JavaScript Required
Streamlit abstracts away the entire frontend stack. Form elements, layouts, theme toggles (Dark/Light mode), and responsive mobile formatting are completely handled automatically.

#### 3. ⏱️ Ultra-Rapid Prototyping
What took days or weeks with traditional web frameworks can be implemented in hours or minutes. Write your script, save the file, and watch the browser auto-refresh instantly.
"""
    )

with col_adv2:
    st.markdown(
        """
#### 4. 🔄 Interactive Reactive Data Applications
Streamlit's execution model turns input widgets into live reactive controls. Changing a filter immediately re-computes visualizations, updates metrics, and slices DataFrames.

#### 5. 🚀 Easy Machine Learning Model Deployment
Load serialized weights (`joblib.load()`, `pickle`, `torch.load()`), provide interactive input sliders or file uploaders, and display live model predictions in real time.

#### 6. 🧰 Rich Suite of Built-in UI Components
Includes sliders, date pickers, multi-select dropdowns, data tables, code blocks with syntax highlighting, LaTeX equations, camera input, audio/video players, and streaming chat bubbles.
"""
    )

st.divider()

# ==============================================================================
# 8. Real-World Applications
# ==============================================================================
st.header("8. Real-World Applications Across Industries", divider="rainbow")
st.markdown(
    "Streamlit is employed across Fortune 500 enterprises, AI startups, and research institutions in the following key domains:"
)

col_app1, col_app2 = st.columns(2)

with col_app1:
    with st.expander("🔬 1. Data Science & Exploratory Data Analysis (EDA)", expanded=True):
        st.markdown(
            """
- **Interactive Dataset Slicers**: Filter gigabyte-scale datasets dynamically by date, category, and numerical thresholds.
- **Missing Value & Outlier Auditing**: Visualize missing data heatmaps and distribution boxplots on demand.
- **Automated EDA Portals**: Upload any CSV or Excel file and automatically generate profiling summaries and correlation matrices.
"""
        )

    with st.expander("🤖 2. Machine Learning Model Evaluation & Prototyping", expanded=True):
        st.markdown(
            """
- **What-If Scenario Analysis**: Adjust feature sliders to see how loan approval probabilities or fraud risk scores change in real time.
- **Confusion Matrix & ROC Curve Inspectors**: Interactive threshold tuning for precision-recall tradeoffs.
- **Hyperparameter Playground**: Tune model parameters (`max_depth`, `learning_rate`) and train models interactively.
"""
        )

    with st.expander("🧠 3. Artificial Intelligence & Computer Vision", expanded=True):
        st.markdown(
            """
- **Image Classification & Segmentation**: Upload images or capture webcam frames with `st.camera_input()` and overlay bounding boxes.
- **Speech-to-Text & Audio Transcription**: Upload audio files, run Whisper inference, and display interactive transcripts.
- **Document OCR**: Extract text and tabular data from uploaded PDFs using Vision models.
"""
        )

    with st.expander("📊 4. Business Analytics & Financial Forecasting", expanded=True):
        st.markdown(
            """
- **Executive KPI Dashboards**: Track Monthly Recurring Revenue (MRR), Customer Acquisition Cost (CAC), and churn metrics.
- **Time-Series Forecasting**: Interactive Prophet or ARIMA forecasting with confidence intervals and seasonality decomposition.
- **Cohort Analysis**: Customer retention and lifetime value (LTV) heatmaps.
"""
        )

with col_app2:
    with st.expander("📈 5. Interactive Data Visualization", expanded=True):
        st.markdown(
            """
- **Geospatial Mapping**: Render 3D geospatial point clouds and vehicle routes using `st.map()` and PyDeck.
- **Plotly & Altair Drilldowns**: Interactive cross-filtering where clicking a bar chart filters an associated scatter plot.
- **Scientific Visualizations**: 3D molecular structures, genome sequence alignments, and engineering telemetry.
"""
        )

    with st.expander("💬 6. Generative AI & LLM Applications", expanded=True):
        st.markdown(
            """
- **Chatbots & Conversational Agents**: Built using native `st.chat_message()` and `st.chat_input()` with token streaming (`st.write_stream`).
- **Retrieval-Augmented Generation (RAG)**: Chat with uploaded PDFs, query internal knowledge bases, and visualize retrieved chunks.
- **Prompt Engineering Playgrounds**: Side-by-side comparison of different system prompts, temperature settings, and model providers.
"""
        )

    with st.expander("🛠️ 7. Internal Business & Operational Tools", expanded=True):
        st.markdown(
            """
- **Customer Support Triage**: Automatically classify incoming support tickets using NLP and route to appropriate teams.
- **Database Query Portals**: Safe, parameterized interfaces for non-technical team members to query internal data warehouses.
- **Automated Report Generators**: Click-to-generate PDF and PowerPoint executive slide decks from live database metrics.
"""
        )

st.divider()

# ==============================================================================
# Summary & Interview Key Takeaways
# ==============================================================================
st.header("🎯 Summary: Key Takeaways for Technical Interviews", divider="rainbow")

st.markdown(
    """
1. **Definition**: Streamlit is an open-source, pure-Python framework that turns data scripts into shareable web apps without HTML, CSS, or JS.
2. **Core Mechanism**: Top-to-bottom reactive script re-execution triggered by user widget interactions.
3. **When to use Streamlit vs Flask/Django**: Choose Streamlit for analytical dashboards, ML demos, and internal tools where development speed is paramount. Choose Flask/FastAPI for REST APIs and Django for complex multi-user transactional SaaS platforms.
4. **When to use Streamlit vs Dash/Gradio**: Choose Streamlit for comprehensive data apps and business dashboards; choose Gradio for rapid model inference demos on Hugging Face; choose Dash when fine-grained callback graph control is strictly required.
5. **Business Value**: Collapses the frontend-backend communication divide, enabling data teams to ship value to stakeholders in minutes.
"""
)
