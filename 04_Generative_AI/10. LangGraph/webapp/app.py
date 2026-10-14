import streamlit as st
import os
import re
from pathlib import Path

st.set_page_config(
    page_title="LangGraph Master Interactive Webapp (Multi-Agent State Machines)",
    page_icon="🕸️",
    layout="wide"
)

def natural_key(string_):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', str(string_))]

def scan_domain_lessons():
    """Scans all chapters and lessons inside this webapp directory."""
    current_dir = Path(__file__).parent
    chapters = [d for d in current_dir.iterdir() if d.is_dir() and not d.name.startswith((".", "__"))]
    chapters.sort(key=lambda p: natural_key(p.name))
    
    nav_dict = {}
    for chap in chapters:
        py_files = [f for f in chap.glob("*.py") if not f.name.startswith("__") and f.stat().st_size > 0]
        py_files.sort(key=lambda f: natural_key(f.name))
        if py_files:
            pages = []
            for pf in py_files:
                title = pf.stem
                pages.append(st.Page(str(pf.relative_to(current_dir)), title=title))
            nav_dict[chap.name] = pages
            
    return nav_dict

nav_structure = scan_domain_lessons()

if nav_structure:
    pg = st.navigation(nav_structure)
    pg.run()
else:
    st.info("🕸️ LangGraph interactive curriculum initialized. Select a chapter and lesson module to begin.")
