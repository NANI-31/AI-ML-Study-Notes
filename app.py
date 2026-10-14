import re
from pathlib import Path
import streamlit as st

# ==============================================================================
# Page Configuration & Master Layout (Wide Mode)
# ==============================================================================
st.set_page_config(
    page_title="AI & ML Study Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Protect against duplicate set_page_config calls in sub-lesson scripts
_orig_set_page_config = st.set_page_config
st.set_page_config = lambda *args, **kwargs: None

# Global responsive styling for wide mode
st.markdown(
    """
    <style>
    .block-container {
        max-width: 96% !important;
        padding-top: 1.5rem !important;
        padding-right: 2rem !important;
        padding-left: 2rem !important;
        padding-bottom: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Master Configuration & Constants
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

TIERS = [
    ("01_Data_Science_Foundations", "Data Science Foundations", "📐", "#3B82F6"),
    ("02_Machine_Learning", "Machine Learning", "🤖", "#10B981"),
    ("03_Data_Visualization", "Data Visualization", "📊", "#F59E0B"),
    ("04_Generative_AI", "Generative AI & LLMs", "✨", "#8B5CF6"),
    ("05_Deployment", "Deployment & Production", "🚀", "#EC4899"),
]

# ==============================================================================
# Helper Utilities
# ==============================================================================
def natural_key(text: str):
    """Key for natural human sorting (e.g. 1.2 before 1.10, 2 before 10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", str(text))]

def extract_leading_number(text: str) -> float:
    """Extract leading number for sorting (e.g. '1.1' -> 1.1, '2' -> 2.0)."""
    match = re.match(r"^(\d+(?:\.\d+)?)", text.strip())
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 999.0
    return 999.0

def get_topic_icon(topic_name: str) -> str:
    """Return an appropriate emoji icon based on topic keyword."""
    name = topic_name.lower()
    if "numpy" in name: return "🔢"
    if "pandas" in name: return "🐼"
    if "spreadsheet" in name or "excel" in name: return "📊"
    if "regex" in name: return "🔍"
    if "sqlite" in name or "sql" in name: return "🗄️"
    if "machine learning" in name or "ml" in name: return "🤖"
    if "sklearn" in name or "scikit" in name: return "⚙️"
    if "data visualization" in name: return "📈"
    if "matplotlib" in name: return "🎨"
    if "seaborn" in name: return "🌊"
    if "plotly" in name: return "📉"
    if "streamlit" in name: return "👑"
    if "prompt" in name: return "✍️"
    if "llm" in name: return "🧠"
    if "embedding" in name: return "🧬"
    if "vector" in name: return "🎯"
    if "chroma" in name: return "🌈"
    if "rag" in name: return "📚"
    if "pydantic" in name: return "🛡️"
    if "agent" in name: return "🕵️"
    if "langchain" in name: return "🦜"
    if "langgraph" in name: return "🕸️"
    if "langsmith" in name: return "🔬"
    if "fastapi" in name: return "⚡"
    if "docker" in name: return "🐳"
    return "📘"

def get_file_icon(filename: str) -> str:
    """Return an intuitive emoji icon based on lesson keywords."""
    name_lower = filename.lower()
    if "what is" in name_lower or "intro" in name_lower: return "⚡"
    if "install" in name_lower: return "📦"
    if "first" in name_lower or "hello" in name_lower: return "🚀"
    if "arch" in name_lower or "diagram" in name_lower: return "🏗️"
    if "text" in name_lower: return "📝"
    if "data" in name_lower or "dataframe" in name_lower: return "📊"
    if "chart" in name_lower or "plot" in name_lower: return "📈"
    if "upload" in name_lower: return "📤"
    if "download" in name_lower: return "📥"
    if "form" in name_lower: return "📋"
    if "widget" in name_lower or "input" in name_lower or "button" in name_lower: return "🎛️"
    if "layout" in name_lower or "container" in name_lower: return "📐"
    if "state" in name_lower or "session" in name_lower: return "💾"
    if "cache" in name_lower: return "⚡"
    if "model" in name_lower or "ml" in name_lower: return "🤖"
    if "chat" in name_lower or "ai" in name_lower or "rag" in name_lower: return "💬"
    if "map" in name_lower: return "🗺️"
    if "media" in name_lower or "image" in name_lower or "video" in name_lower: return "🎬"
    return "📄"

# ==============================================================================
# Dynamic Catalog Scanner (Fast Cached Discovery with 5s TTL)
# ==============================================================================
@st.cache_data(ttl=5, show_spinner=False)
def scan_repository():
    """Discover all tiers, topics, webapps, lesson files, and syllabi across the repo."""
    catalog = []
    
    for tier_dir_name, tier_title, tier_icon, tier_color in TIERS:
        tier_path = BASE_DIR / tier_dir_name
        if not tier_path.exists():
            continue
            
        topics = []
        subdirs = [p for p in tier_path.iterdir() if p.is_dir() and not p.name.startswith((".", "__"))]
        subdirs.sort(key=lambda p: natural_key(p.name))
        
        for topic_path in subdirs:
            webapp_dir = topic_path / "webapp"
            py_files = []
            sections_dict = {}
            
            if webapp_dir.exists():
                # Scan subdirectories inside webapp (chapters)
                chap_dirs = [d for d in webapp_dir.iterdir() if d.is_dir() and not d.name.startswith((".", "__")) and d.name != "demo"]
                chap_dirs.sort(key=lambda d: natural_key(d.name))
                
                for c_dir in chap_dirs:
                    # Count non-empty Python lesson files
                    c_py_files = [f for f in c_dir.glob("*.py") if not f.name.startswith("__") and f.stat().st_size > 0]
                    c_py_files.sort(key=lambda f: natural_key(f.name))
                    if c_py_files:
                        sections_dict[c_dir.name] = c_py_files
                        py_files.extend(c_py_files)
                
                # Check top-level python files in webapp (excluding root app.py)
                root_lessons = [f for f in webapp_dir.glob("*.py") if f.name != "app.py" and not f.name.startswith("__") and f.stat().st_size > 0]
                root_lessons.sort(key=lambda f: natural_key(f.name))
                if root_lessons:
                    sections_dict["General Lessons"] = root_lessons
                    py_files.extend(root_lessons)

            # Discover roadmap documents
            roadmap_txt = list(topic_path.glob("*.txt"))
            roadmap_md = [f for f in topic_path.glob("*prompt*.md")]
            learn_dir = topic_path / "learn"
            learn_files = list(learn_dir.glob("*.*")) if learn_dir.exists() else []
            
            topic_id = f"{tier_dir_name}/{topic_path.name}"
            topics.append({
                "id": topic_id,
                "name": topic_path.name,
                "tier_dir": tier_dir_name,
                "tier_title": tier_title,
                "tier_icon": tier_icon,
                "tier_color": tier_color,
                "path": str(topic_path),
                "webapp_dir": str(webapp_dir),
                "py_files": [str(f) for f in py_files],
                "lesson_count": len(py_files),
                "sections_dict": {k: [str(f) for f in v] for k, v in sections_dict.items()},
                "roadmap_txt": [str(f) for f in roadmap_txt],
                "roadmap_md": [str(f) for f in roadmap_md],
                "has_learn": len(learn_files) > 0,
                "learn_count": len(learn_files),
            })
            
        catalog.append({
            "dir_name": tier_dir_name,
            "title": tier_title,
            "icon": tier_icon,
            "color": tier_color,
            "topics": topics,
        })
        
    return catalog

# Load catalog
CATALOG = scan_repository()
ALL_TOPICS = [t for tier in CATALOG for t in tier["topics"]]
TOPIC_LOOKUP = {t["id"]: t for t in ALL_TOPICS}
NAME_TO_ID = {t["name"]: t["id"] for t in ALL_TOPICS}

# Build global slug reverse lookup to automatically recover roadmap context on page refresh
SLUG_TO_TOPIC_ID = {}
for t in ALL_TOPICS:
    tid = t["id"]
    for sec_name, p_paths in t["sections_dict"].items():
        for p_str in p_paths:
            stem = Path(p_str).stem
            slug = re.sub(r"[^a-zA-Z0-9_-]", "-", stem.lower()).strip("-")
            SLUG_TO_TOPIC_ID[slug] = tid

# ==============================================================================
# Session State & URL Routing Recovery
# ==============================================================================
def clear_script_intent():
    """Clear Streamlit internal script intent so navigation transitions cleanly."""
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
        ctx = get_script_run_ctx(suppress_warning=True)
        if ctx and ctx.pages_manager:
            ctx.pages_manager.set_script_intent("", "")
    except Exception:
        pass

def get_intended_page_slug() -> str | None:
    """Retrieve the URL slug requested by the browser on direct navigation or refresh."""
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
        ctx = get_script_run_ctx(suppress_warning=True)
        if ctx and ctx.pages_manager and ctx.pages_manager.intended_page_name:
            return ctx.pages_manager.intended_page_name
    except Exception:
        pass
    return None

if "active_topic_id" not in st.session_state:
    st.session_state.active_topic_id = None
if "selected_tier_filter" not in st.session_state:
    st.session_state.selected_tier_filter = "All"
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# Dual-layer state recovery:
# Layer 1: Check URL query parameter '?topic=...'
url_topic_id = st.query_params.get("topic")

# Layer 2: Check requested page slug via intended_page_name
intended_slug = get_intended_page_slug()

# Recover topic ID if not active in session state (e.g. browser refresh or direct link)
if not st.session_state.active_topic_id:
    if url_topic_id and url_topic_id in TOPIC_LOOKUP:
        st.session_state.active_topic_id = url_topic_id
    elif intended_slug and intended_slug in SLUG_TO_TOPIC_ID:
        st.session_state.active_topic_id = SLUG_TO_TOPIC_ID[intended_slug]

# Synchronize URL query parameter to reflect active topic
if st.session_state.active_topic_id:
    if st.query_params.get("topic") != st.session_state.active_topic_id:
        st.query_params["topic"] = st.session_state.active_topic_id

# ==============================================================================
# View: Master Study Portal (Home Landing)
# ==============================================================================
def render_master_portal():
    # Hero Banner
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 2.2rem; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 2rem;">
            <h1 style="color: #f8fafc; margin-bottom: 0.5rem; font-size: 2.5rem; font-weight: 800;">
                🎓 AI & Machine Learning Knowledge Hub
            </h1>
            <p style="color: #94a3b8; font-size: 1.15rem; margin-bottom: 0;">
                Master Study Portal • Interactive Curriculums • Applied WebApps • Code Sandboxes
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Repository Statistics Metrics
    total_tiers = len(CATALOG)
    total_topics = len(ALL_TOPICS)
    total_lessons = sum(t["lesson_count"] for t in ALL_TOPICS)
    ready_topics = sum(1 for t in ALL_TOPICS if t["lesson_count"] > 0)
    syllabus_topics = sum(1 for t in ALL_TOPICS if len(t["roadmap_txt"]) > 0 or len(t["roadmap_md"]) > 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Learning Domains", total_tiers, "Core Pillars")
    col2.metric("Curriculums", total_topics, "Comprehensive")
    col3.metric("Live Interactive Lessons", total_lessons, f"{ready_topics} Apps Active")
    col4.metric("Syllabi & Roadmaps Ready", syllabus_topics, "Structured")

    st.divider()

    # Search & Tier Filters
    col_filter1, col_filter2 = st.columns([2, 1])
    
    with col_filter1:
        tier_options = ["All"] + [tier["title"] for tier in CATALOG]
        selected_tier = st.segmented_control(
            "Filter by Domain:",
            options=tier_options,
            default=st.session_state.selected_tier_filter,
            key="tier_filter_widget"
        )
        if selected_tier:
            st.session_state.selected_tier_filter = selected_tier
            
    with col_filter2:
        search_query = st.text_input("🔍 Search topic or keyword:", value="", placeholder="e.g. Numpy, Plotly, LLM...")

    st.write("")

    # Filtered Topics Display
    for tier in CATALOG:
        # Check tier filter
        if st.session_state.selected_tier_filter != "All" and tier["title"] != st.session_state.selected_tier_filter:
            continue
            
        # Check search query
        matching_topics = []
        for t in tier["topics"]:
            if search_query:
                q = search_query.lower()
                if q in t["name"].lower() or q in tier["title"].lower():
                    matching_topics.append(t)
            else:
                matching_topics.append(t)
                
        if not matching_topics:
            continue

        # Render Tier Section Header
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: 1.5rem; margin-bottom: 0.75rem;">
                <span style="font-size: 1.8rem;">{tier["icon"]}</span>
                <h2 style="margin: 0; font-size: 1.6rem; color: #f1f5f9;">{tier["title"]}</h2>
                <span style="background-color: {tier['color']}22; color: {tier['color']}; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.85rem; font-weight: 600; border: 1px solid {tier['color']}44;">
                    {len(matching_topics)} Roadmaps
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2-column or 3-column card layout
        cols = st.columns(3)
        for idx, topic in enumerate(matching_topics):
            col = cols[idx % 3]
            topic_icon = get_topic_icon(topic["name"])
            
            has_lessons = topic["lesson_count"] > 0
            has_roadmap = len(topic["roadmap_txt"]) > 0 or len(topic["roadmap_md"]) > 0
            
            with col:
                with st.container(border=True):
                    # Topic header
                    st.markdown(f"### {topic_icon} {topic['name']}")
                    
                    # Badges & status
                    if has_lessons:
                        st.markdown(
                            f"<span style='background: #10B98122; color: #10B981; padding: 0.2rem 0.5rem; border-radius: 6px; font-weight: 600; font-size: 0.8rem; border: 1px solid #10B98144;'>🟢 {topic['lesson_count']} Live Lessons</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption(f"{len(topic['sections_dict'])} modules with interactive Streamlit tutorials")
                        btn_label = f"🚀 Launch {topic['name']} App"
                        btn_type = "primary"
                    elif has_roadmap:
                        st.markdown(
                            f"<span style='background: #F59E0B22; color: #F59E0B; padding: 0.2rem 0.5rem; border-radius: 6px; font-weight: 600; font-size: 0.8rem; border: 1px solid #F59E0B44;'>🟡 Syllabus & Roadmap Ready</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption("Detailed curriculum outline, prompts, and study benchmarks")
                        btn_label = f"📖 Open Roadmap & Notes"
                        btn_type = "secondary"
                    else:
                        st.markdown(
                            f"<span style='background: #64748B22; color: #94A3B8; padding: 0.2rem 0.5rem; border-radius: 6px; font-weight: 600; font-size: 0.8rem; border: 1px solid #64748B44;'>⚪ Scaffold Ready</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption("Module folder initialized and ready for lessons")
                        btn_label = f"🛠️ Explore Module"
                        btn_type = "secondary"

                    st.write("")
                    if st.button(btn_label, key=f"btn_card_{topic['id']}", use_container_width=True, type=btn_type):
                        clear_script_intent()
                        st.session_state.active_topic_id = topic["id"]
                        st.query_params["topic"] = topic["id"]
                        st.rerun()

    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: #64748b; padding: 1.5rem;">
            Built with <b>Streamlit</b> • Unified Multi-Roadmap Interactive Learning Architecture
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# View: Topic Syllabus & Roadmap Studio (For Roadmaps without Live Lessons yet)
# ==============================================================================
def render_topic_syllabus_studio(topic: dict):
    topic_icon = get_topic_icon(topic["name"])
    
    # Top Action Bar: Return or switch
    col_nav1, col_nav2 = st.columns([1, 4])
    with col_nav1:
        if st.button("⬅️ All Roadmaps", key="top_back_btn_syll", use_container_width=True, type="primary"):
            clear_script_intent()
            st.session_state.active_topic_id = None
            if "topic" in st.query_params:
                del st.query_params["topic"]
            st.rerun()
    with col_nav2:
        st.caption(f"{topic['tier_icon']} {topic['tier_title']} / **{topic['name']}**")

    st.title(f"{topic_icon} {topic['name']} — Study Roadmap")
    st.markdown(f"**Domain:** {topic['tier_icon']} `{topic['tier_title']}`")

    st.divider()

    # Load Roadmap Text if available
    txt_content = ""
    if topic["roadmap_txt"]:
        try:
            txt_path = Path(topic["roadmap_txt"][0])
            txt_content = txt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            txt_content = f"Error reading roadmap text: {e}"

    md_content = ""
    if topic["roadmap_md"]:
        try:
            md_path = Path(topic["roadmap_md"][0])
            md_content = md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            md_content = f"Error reading prompt markdown: {e}"

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📑 Structured Curriculum",
        "📄 Complete Roadmap Text",
        "💡 System Prompts & Tests",
        "🛠️ Interactive WebApp Builder"
    ])

    with tab1:
        if txt_content:
            st.subheader("Curriculum Chapters & Milestone Checklist", divider="blue")
            
            # Parse top chapters (lines starting with digits and uppercase or letters)
            lines = txt_content.splitlines()
            chapters = []
            current_chapter = None
            current_lines = []
            
            for line in lines:
                stripped = line.strip()
                # Check for major chapter: e.g. '1. Introduction...', '2. Basics...'
                m = re.match(r"^(\d+)\.\s+([A-Za-z].*)", line)
                if line and line[0].isdigit() and m:
                    if current_chapter:
                        chapters.append((current_chapter, "\n".join(current_lines)))
                    current_chapter = stripped
                    current_lines = []
                else:
                    if current_chapter:
                        current_lines.append(line)
            if current_chapter:
                chapters.append((current_chapter, "\n".join(current_lines)))

            if chapters:
                st.caption(f"Parsed {len(chapters)} major chapters from `{Path(topic['roadmap_txt'][0]).name}`")
                
                # Progress calculation
                checked_count = 0
                total_chapters = len(chapters)
                
                for idx, (chap_title, chap_body) in enumerate(chapters):
                    with st.expander(f"📌 {chap_title}", expanded=(idx < 2)):
                        # Interactive check box for study tracking
                        ch_key = f"chk_{topic['id']}_{idx}"
                        is_done = st.checkbox("Mark chapter completed", key=ch_key)
                        if is_done:
                            checked_count += 1
                        
                        st.markdown("```text\n" + chap_body.strip() + "\n```")
                        
                st.sidebar.progress(checked_count / max(total_chapters, 1), text=f"Progress: {checked_count}/{total_chapters} Chapters")
            else:
                st.markdown(txt_content)
        else:
            st.info("No `.txt` roadmap outline detected yet for this topic. You can add one anytime!")

    with tab2:
        if txt_content:
            st.subheader("Full Roadmap Reference", divider="gray")
            st.download_button(
                "📥 Download Roadmap Outline",
                data=txt_content,
                file_name=f"{topic['name']}_roadmap.txt",
                mime="text/plain"
            )
            st.code(txt_content, language="markdown")
        else:
            st.warning("No roadmap outline file found.")

    with tab3:
        if md_content:
            st.subheader("Prompt Engineering & Practice Tests", divider="rainbow")
            st.markdown(md_content)
        else:
            st.info("No `_prompt.md` file found for this topic.")

    with tab4:
        st.subheader("Build Interactive Lessons for this Module", divider="green")
        st.markdown(
            f"""
            This module currently has **0 interactive Python lessons** inside its `webapp/` directory:
            `{topic['webapp_dir']}`
            
            ### How to Add Lessons to this Roadmap:
            1. Create chapter folders inside `{topic['name']}/webapp/` (e.g. `1. Basics`, `2. Core Functions`).
            2. Add `.py` Streamlit files (e.g. `1.1 Intro.py`, `1.2 Tutorial.py`).
            3. **Save and reload**: As soon as any `.py` file is saved, this master portal automatically registers it into an interactive multi-page tutorial application!
            """
        )

# ==============================================================================
# View: Active Interactive Roadmap Overview Dashboard
# ==============================================================================
def render_active_roadmap_overview(topic: dict):
    topic_icon = get_topic_icon(topic["name"])
    
    # Top Action Bar: Return to all roadmaps
    col_nav1, col_nav2 = st.columns([1, 4])
    with col_nav1:
        if st.button("⬅️ All Roadmaps", key="top_back_btn_overview", use_container_width=True, type="primary"):
            clear_script_intent()
            st.session_state.active_topic_id = None
            if "topic" in st.query_params:
                del st.query_params["topic"]
            st.rerun()
    with col_nav2:
        st.caption(f"{topic['tier_icon']} {topic['tier_title']} / **{topic['name']}**")

    st.title(f"{topic_icon} {topic['name']} Interactive Learning Portal")
    st.markdown(f"**Domain:** {topic['tier_icon']} `{topic['tier_title']}`")

    st.divider()

    # Load Roadmap Text & Prompt MD if available
    txt_content = ""
    if topic["roadmap_txt"]:
        try:
            txt_path = Path(topic["roadmap_txt"][0])
            txt_content = txt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            txt_content = f"Error reading roadmap text: {e}"

    md_content = ""
    if topic["roadmap_md"]:
        try:
            md_path = Path(topic["roadmap_md"][0])
            md_content = md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            md_content = f"Error reading prompt markdown: {e}"

    tab_lessons, tab_roadmap, tab_prompts = st.tabs([
        "🚀 Interactive Lessons & Tutorials",
        "📑 Curriculum Roadmap & Checklist",
        "💡 System Prompts & Tests",
    ])

    with tab_lessons:
        # Metrics
        total_sections = len(topic["sections_dict"])
        total_lessons = topic["lesson_count"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Course Sections", total_sections)
        col2.metric("Interactive Python Lessons", total_lessons)
        col3.metric("Platform Status", "🟢 Ready & Interactive")

        st.divider()
        st.subheader("🗂️ Curriculum Modules & Interactive Lessons", divider="rainbow")

        for section_name, py_file_paths in topic["sections_dict"].items():
            with st.container(border=True):
                st.markdown(f"### 📁 {section_name}")
                st.caption(f"Contains {len(py_file_paths)} interactive lessons")

                grid_cols = st.columns(2)
                for idx, p_path_str in enumerate(py_file_paths):
                    p_path = Path(p_path_str)
                    target_col = grid_cols[idx % 2]
                    lesson_title = p_path.stem
                    icon = get_file_icon(lesson_title)
                    
                    with target_col:
                        btn_label = f"{icon} {lesson_title}"
                        page_obj = ST_PAGE_LOOKUP.get(str(p_path.resolve()))
                        if page_obj and st.button(btn_label, key=f"dash_btn_{p_path.stem}", use_container_width=True, type="primary"):
                            st.switch_page(page_obj, query_params={"topic": active_topic["id"]})

    with tab_roadmap:
        if txt_content:
            st.subheader("Curriculum Chapters & Milestone Checklist", divider="blue")
            lines = txt_content.splitlines()
            chapters = []
            current_chapter = None
            current_lines = []
            for line in lines:
                stripped = line.strip()
                m = re.match(r"^(\d+)\.\s+([A-Za-z].*)", line)
                if line and line[0].isdigit() and m:
                    if current_chapter:
                        chapters.append((current_chapter, "\n".join(current_lines)))
                    current_chapter = stripped
                    current_lines = []
                else:
                    if current_chapter:
                        current_lines.append(line)
            if current_chapter:
                chapters.append((current_chapter, "\n".join(current_lines)))

            if chapters:
                for idx, (chap_title, chap_body) in enumerate(chapters):
                    with st.expander(f"📌 {chap_title}", expanded=(idx < 2)):
                        st.markdown("```text\n" + chap_body.strip() + "\n```")
            else:
                st.markdown(txt_content)
        else:
            st.info("No roadmap outline text found.")

    with tab_prompts:
        if md_content:
            st.markdown(md_content)
        else:
            st.info("No prompt template file found.")

# ==============================================================================
# Dynamic Navigation Router Construction
# ==============================================================================
active_topic = TOPIC_LOOKUP.get(st.session_state.active_topic_id)
ST_PAGE_LOOKUP = {}

# CASE 1: Master Portal Mode (No Active Roadmap Selected)
if active_topic is None:
    home_page = st.Page(
        render_master_portal,
        title="Master Study Hub",
        icon="🌐",
        default=True
    )
    home_hub_page = st.Page(
        render_master_portal,
        title="All Roadmaps",
        icon="🌐",
        url_path="hub"
    )
    current_page = st.navigation([home_page, home_hub_page], position="hidden")
    current_page.run()

# CASE 2: Active Roadmap Mode
else:
    topic_icon = get_topic_icon(active_topic["name"])
    
    # Sub-case 2A: The topic has interactive .py lessons (e.g. Streamlit)
    if active_topic["lesson_count"] > 0:
        # 1. Create Roadmap Overview Page & Home Navigation Page
        overview_page = st.Page(
            lambda: render_active_roadmap_overview(active_topic),
            title=f"Course Overview",
            icon="🏠",
            default=True
        )
        home_nav_page = st.Page(
            render_master_portal,
            title="All Roadmaps",
            icon="🌐",
            url_path="hub"
        )
        
        # 2. Build sections dictionary for navigation
        nav_dict = {
            f"{topic_icon} {active_topic['name']}": [overview_page]
        }
        
        for section_name, py_path_strings in active_topic["sections_dict"].items():
            section_pages = []
            for p_str in py_path_strings:
                p_path = Path(p_str)
                lesson_title = p_path.stem
                icon = get_file_icon(lesson_title)
                url_slug = re.sub(r"[^a-zA-Z0-9_-]", "-", lesson_title.lower()).strip("-")
                
                lesson_page = st.Page(
                    str(p_path),
                    title=lesson_title,
                    icon=icon,
                    url_path=url_slug
                )
                section_pages.append(lesson_page)
                ST_PAGE_LOOKUP[str(p_path.resolve())] = lesson_page
                
            nav_dict[f"📁 {section_name}"] = section_pages

        # Use modern Streamlit navigation with custom sidebar controls
        all_pages_list = [overview_page, home_nav_page] + [p for pages in nav_dict.values() for p in pages if p != overview_page]
        current_page = st.navigation(all_pages_list, position="hidden")

        # Custom Collapsible & Persistent Sidebar
        with st.sidebar:
            st.markdown(f"### {topic_icon} {active_topic['name']}")
            st.caption(f"{active_topic['tier_icon']} {active_topic['tier_title']}")

            # Button to return to the Master Portal
            if st.button("⬅️ Back to All Roadmaps", key="nav_back_to_portal", use_container_width=True, type="primary"):
                clear_script_intent()
                st.session_state.active_topic_id = None
                if "topic" in st.query_params:
                    del st.query_params["topic"]
                st.switch_page(home_nav_page, query_params={})

            st.divider()
            
            # Quick Switcher Dropdown to jump across any roadmap directly
            topic_names = [t["name"] for t in ALL_TOPICS]
            curr_idx = topic_names.index(active_topic["name"])
            selected_switch = st.selectbox(
                "⚡ Switch Roadmap:",
                topic_names,
                index=curr_idx,
                key="sidebar_topic_switcher"
            )
            if selected_switch != active_topic["name"]:
                clear_script_intent()
                new_id = NAME_TO_ID[selected_switch]
                st.session_state.active_topic_id = new_id
                st.query_params["topic"] = new_id
                st.rerun()

            st.divider()

            # Overview Navigation Button
            is_ov_active = (current_page == overview_page)
            if st.button("🏠 Roadmap Overview", key="nav_overview_btn", use_container_width=True, type="primary" if is_ov_active else "secondary"):
                st.switch_page(overview_page, query_params={"topic": active_topic["id"]})

            st.divider()
            st.markdown("##### 📚 Curriculum Chapters")

            # Render collapsible section expanders
            for sec_header, pages in nav_dict.items():
                if sec_header.startswith(topic_icon):
                    continue
                is_current_sec = any(current_page == p for p in pages)
                with st.expander(sec_header, expanded=is_current_sec):
                    for p in pages:
                        is_active = (current_page == p)
                        btn_label = f"{p.icon} {p.title}" if p.icon else f"📄 {p.title}"
                        if st.button(
                            btn_label,
                            key=f"nav_btn_{p.url_path}",
                            use_container_width=True,
                            type="primary" if is_active else "secondary"
                        ):
                            st.switch_page(p, query_params={"topic": active_topic["id"]})

        # Run the selected page
        current_page.run()

    # Sub-case 2B: The topic does NOT have .py lessons yet (Roadmap & Syllabus Studio Mode)
    else:
        syllabus_page = st.Page(
            lambda: render_topic_syllabus_studio(active_topic),
            title=f"{active_topic['name']} Roadmap",
            icon="📋",
            default=True
        )
        current_page = st.navigation([syllabus_page], position="hidden")
        
        # Sidebar with switcher and back button
        with st.sidebar:
            st.markdown(f"### {topic_icon} {active_topic['name']}")
            st.caption(f"{active_topic['tier_icon']} {active_topic['tier_title']}")

            if st.button("⬅️ Back to All Roadmaps", key="nav_back_to_portal_syll", use_container_width=True, type="primary"):
                clear_script_intent()
                st.session_state.active_topic_id = None
                if "topic" in st.query_params:
                    del st.query_params["topic"]
                st.rerun()

            st.divider()
            
            topic_names = [t["name"] for t in ALL_TOPICS]
            curr_idx = topic_names.index(active_topic["name"])
            selected_switch = st.selectbox(
                "⚡ Switch Roadmap:",
                topic_names,
                index=curr_idx,
                key="sidebar_topic_switcher_syll"
            )
            if selected_switch != active_topic["name"]:
                clear_script_intent()
                new_id = NAME_TO_ID[selected_switch]
                st.session_state.active_topic_id = new_id
                st.query_params["topic"] = new_id
                st.rerun()

        current_page.run()
