import re
from pathlib import Path
import streamlit as st

# ==============================================================================
# Helper Functions for File & Directory Discovery
# ==============================================================================
BASE_DIR = Path(__file__).parent.resolve()

def extract_leading_number(text: str) -> float:
    """Extract leading number for natural sorting (e.g., '1.1' -> 1.1, '2' -> 2.0)."""
    match = re.match(r"^(\d+(?:\.\d+)?)", text.strip())
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 999.0
    return 999.0

def get_file_icon(filename: str) -> str:
    """Return an intuitive emoji icon based on lesson topic keywords."""
    name_lower = filename.lower()
    if "what is" in name_lower or "intro" in name_lower:
        return "⚡"
    if "install" in name_lower:
        return "📦"
    if "first" in name_lower or "hello" in name_lower:
        return "🚀"
    if "arch" in name_lower:
        return "🏗️"
    if "text" in name_lower:
        return "📝"
    if "data" in name_lower or "dataframe" in name_lower:
        return "📊"
    if "chart" in name_lower or "plot" in name_lower:
        return "📈"
    if "upload" in name_lower:
        return "📤"
    if "download" in name_lower:
        return "📥"
    if "file" in name_lower:
        return "📁"
    if "form" in name_lower:
        return "📋"
    if "widget" in name_lower or "input" in name_lower:
        return "🎛️"
    if "layout" in name_lower or "container" in name_lower:
        return "📐"
    if "state" in name_lower:
        return "💾"
    if "cache" in name_lower:
        return "⚡"
    if "chat" in name_lower or "ai" in name_lower:
        return "💬"
    if "diagram" in name_lower or "showcase" in name_lower or "arch" in name_lower:
        return "🗺️"
    return "📄"

# ==============================================================================
# Dynamic Curriculum Discovery
# ==============================================================================
sections_data = {}
all_pages = []
page_lookup = {}

# Discover all section directories (e.g. '1. Introduction to Streamlit', '2. Streamlit Basics')
subdirs = [
    d for d in BASE_DIR.iterdir()
    if d.is_dir()
    and not d.name.startswith(".")
    and not d.name.startswith("__")
    and d.name not in ["demo"]
]
subdirs.sort(key=lambda d: (extract_leading_number(d.name), d.name))

for folder in subdirs:
    py_files = [
        f for f in folder.glob("*.py")
        if not f.name.startswith("__")
    ]
    py_files.sort(key=lambda f: (extract_leading_number(f.name), f.name))

    if not py_files:
        continue

    folder_pages = []
    for py_file in py_files:
        lesson_title = py_file.stem
        icon = get_file_icon(lesson_title)
        
        # Create an st.Page for each python lesson file
        page = st.Page(
            str(py_file),
            title=lesson_title,
            icon=icon,
            url_path=re.sub(r"[^a-zA-Z0-9_-]", "-", lesson_title.lower()).strip("-"),
        )
        folder_pages.append(page)
        page_lookup[str(py_file.resolve())] = page
        all_pages.append(page)

    sections_data[folder.name] = folder_pages

# ==============================================================================
# Home / Dashboard Landing View
# ==============================================================================
def render_home_view():
    st.title("📚 Streamlit Interactive Learning Hub")
    st.markdown(
        """
Welcome to the master portal for the **Streamlit Interactive Curriculum**.
Use the **collapsible sections in the left sidebar** to navigate between any module,
or launch individual lessons directly from the curriculum cards below.
"""
    )

    st.divider()

    # High-level Statistics
    total_sections = len(sections_data)
    total_lessons = sum(len(pages) for pages in sections_data.values())

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("Discovered Sections", total_sections)
    col_stat2.metric("Total Python Lessons", total_lessons)
    col_stat3.metric("Streamlit Version", st.__version__)

    st.divider()

    st.subheader("🗂️ Curriculum Modules & Lessons", divider="rainbow")

    for section_name, pages in sections_data.items():
        folder_icon = "🎲" if "random" in section_name.lower() else "📁"
        with st.container(border=True):
            st.markdown(f"### {folder_icon} {section_name}")
            st.caption(f"Contains {len(pages)} interactive lesson files")

            grid_cols = st.columns(2)
            for idx, p in enumerate(pages):
                target_col = grid_cols[idx % 2]
                with target_col:
                    btn_label = f"{p.icon} {p.title}" if p.icon else f"📄 {p.title}"
                    if st.button(
                        btn_label,
                        key=f"home_btn_{p.url_path}",
                        use_container_width=True,
                    ):
                        st.switch_page(p)

    st.divider()
    st.markdown("### 💡 How It Works")
    st.markdown(
        """
- **Dynamic File Scanning**: Any new `.py` files or numbered folders added to this directory will appear automatically in the sidebar and home dashboard.
- **Collapsible Sidebar**: Click on any section header (e.g. `📁 1. Introduction to Streamlit`) to expand or collapse that section.
- **Hot-Reload Active**: Any edits made to any lesson file (`Ctrl + S`) automatically update in your browser.
"""
    )

home_page = st.Page(
    render_home_view,
    title="Home / Overview",
    icon="🏠",
    default=True,
)

# ==============================================================================
# Navigation Setup & Custom Collapsible Sidebar
# ==============================================================================
# Register all pages with Streamlit's router with hidden native menu
navigation_list = [home_page] + all_pages
current_page = st.navigation(navigation_list, position="hidden")

# Render Custom Collapsible Sidebar
with st.sidebar:
    st.markdown("### 📚 Streamlit Hub")
    st.caption("Interactive Study Notes")

    # Home Navigation Button
    is_home_active = (current_page == home_page)
    if st.button(
        "🏠 Curriculum Overview",
        key="nav_btn_home",
        use_container_width=True,
        type="primary" if is_home_active else "secondary",
    ):
        st.switch_page(home_page)

    st.divider()
    st.markdown("##### 📁 Curriculum Sections")

    # Render each section as an individual collapsible expander
    for section_name, pages in sections_data.items():
        # Auto-expand the folder containing the currently viewed page
        is_current_section = any(current_page == p for p in pages)
        folder_icon = "🎲" if "random" in section_name.lower() else "📁"

        with st.expander(f"{folder_icon} {section_name}", expanded=is_current_section):
            for p in pages:
                is_active = (current_page == p)
                btn_label = f"{p.icon} {p.title}" if p.icon else f"📄 {p.title}"

                if st.button(
                    btn_label,
                    key=f"sidebar_btn_{p.url_path}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.switch_page(p)

# Run the selected page
current_page.run()
