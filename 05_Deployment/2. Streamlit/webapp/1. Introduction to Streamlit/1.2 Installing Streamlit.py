import sys
import platform
import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="1.2 Installing Streamlit | Complete Guide",
    page_icon="📦",
    layout="wide",
)

st.title("📦 1.2 Installing Streamlit — Environment Setup, Dependencies & Project Structure")
st.markdown(
    """
Welcome to **Lesson 1.2: Installing Streamlit**.
This module provides a complete, step-by-step technical guide to setting up Python,
installing Streamlit across `pip` and `conda`, creating virtual environments, verifying installations,
understanding underlying dependencies, and structuring professional project directories.
"""
)

st.divider()

# ==============================================================================
# 1. Installing Python
# ==============================================================================
st.header("1. Installing Python", divider="rainbow")

col_py1, col_py2 = st.columns([3, 2])

with col_py1:
    st.markdown(
        """
### Prerequisites & Supported Versions
- Streamlit requires **Python 3.8 to 3.12+** (Python 3.11 or 3.12 is highly recommended for best performance and stability).
- Always ensure Python is added to your system's `PATH` environment variable during installation.

#### Platform-Specific Installation Steps:
1. **Windows**:
   - Download the official 64-bit installer from [python.org/downloads](https://www.python.org/downloads/).
   - ⚠️ **Critical Step**: Check the box **"Add python.exe to PATH"** on the first installer screen before clicking *Install Now*.
   - Or install via Windows Package Manager:
     ```powershell
     winget install Python.Python.3.12
     ```
2. **macOS**:
   - Install using Homebrew:
     ```bash
     brew install python@3.12
     ```
3. **Linux (Ubuntu / Debian)**:
   - Install via `apt` package manager:
     ```bash
     sudo apt update
     sudo apt install -y python3 python3-pip python3-venv
     ```
"""
    )

with col_py2:
    st.markdown("#### 🔍 Live System Environment Diagnostic")
    with st.container(border=True):
        st.markdown("**Your Current Host System:**")
        st.write(f"- **Python Version:** `{platform.python_version()}`")
        st.write(f"- **Operating System:** `{platform.system()} ({platform.release()})`")
        st.write(f"- **Architecture:** `{platform.machine()}`")
        st.write(f"- **Executable Path:** `{sys.executable}`")
        if sys.version_info >= (3, 8):
            st.success("✅ System Python meets Streamlit requirements (≥ 3.8).")
        else:
            st.error("🚨 Python version is below 3.8. Please upgrade.")

st.divider()

# ==============================================================================
# 2. Installing Streamlit using pip
# ==============================================================================
st.header("2. Installing Streamlit Using `pip`", divider="rainbow")

st.markdown(
    """
`pip` is Python's default package installer and the standard method for installing Streamlit from the Python Package Index (PyPI).
"""
)

col_pip1, col_pip2 = st.columns(2)

with col_pip1:
    st.markdown("#### 💻 Standard Installation Commands")
    st.code(
        """# 1. Standard installation of latest stable release
pip install streamlit

# 2. Upgrade an existing installation to latest version
pip install --upgrade streamlit

# 3. Install a specific version for reproducibility
pip install streamlit==1.40.0

# 4. Install with optional cloud/enterprise extras (e.g. Snowflake)
pip install "streamlit[snowflake]" """,
        language="bash",
    )

with col_pip2:
    st.markdown("#### ⚙️ Common Flags & Best Practices")
    st.markdown(
        """
- **`--upgrade` (`-U`)**: Fetches the newest Streamlit release from PyPI.
- **`--user`**: Installs into your user profile directory when you lack administrative privileges.
- **`--no-cache-dir`**: Forces downloading clean wheels instead of using cached local files (useful when debugging broken installations).
- **Verify pip is updated**:
  ```bash
  python -m pip install --upgrade pip
  ```
"""
    )

st.divider()

# ==============================================================================
# 3. Installing Streamlit in a Virtual Environment
# ==============================================================================
st.header("3. Installing Streamlit in an Isolated Virtual Environment", divider="rainbow")

st.markdown(
    """
> **Best Practice Rule**: Never install Streamlit into your global system Python. Always create a **virtual environment** for each project to isolate dependencies and prevent library version collisions.
"""
)

col_venv1, col_venv2 = st.columns(2)

with col_venv1:
    st.markdown("#### 🪟 Windows (PowerShell / CMD)")
    st.code(
        """# Step 1: Navigate to project directory
cd "d:\\Data\\GitHub\\my_project"

# Step 2: Create a virtual environment named .venv
python -m venv .venv

# Step 3: Activate the virtual environment
# In PowerShell:
.\\.venv\\Scripts\\Activate.ps1
# In CMD:
.\\.venv\\Scripts\\activate.bat

# Step 4: Install Streamlit inside .venv
pip install streamlit

# Step 5: Deactivate when finished
deactivate""",
        language="powershell",
    )

with col_venv2:
    st.markdown("#### 🍎 macOS & 🐧 Linux (Bash / Zsh)")
    st.code(
        """# Step 1: Navigate to project directory
cd ~/projects/my_project

# Step 2: Create a virtual environment named .venv
python3 -m venv .venv

# Step 3: Activate the virtual environment
source .venv/bin/activate

# Step 4: Install Streamlit inside .venv
pip install streamlit

# Step 5: Deactivate when finished
deactivate""",
        language="bash",
    )

with st.expander("⚠️ Windows Troubleshooting: 'Execution of scripts is disabled on this system'"):
    st.markdown(
        r"""
If PowerShell displays:
`File ...\Activate.ps1 cannot be loaded because running scripts is disabled on this system`

Run this one-time command in PowerShell to grant script permissions for your user account:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
"""
    )

st.divider()

# ==============================================================================
# 4. Installing Streamlit using Conda
# ==============================================================================
st.header("4. Installing Streamlit Using Conda (Anaconda / Miniconda)", divider="rainbow")

st.markdown(
    """
Conda is widely used in data science environments because it manages both Python packages and low-level binary C/C++ dependencies.
"""
)

col_conda1, col_conda2 = st.columns(2)

with col_conda1:
    st.markdown("#### 🐍 Conda Workflow")
    st.code(
        """# Step 1: Create a dedicated environment with specific Python version
conda create -n streamlit_env python=3.11 -y

# Step 2: Activate the environment
conda activate streamlit_env

# Step 3: Install Streamlit from the community-maintained conda-forge channel
conda install -c conda-forge streamlit -y

# Alternatively, you can use pip inside the activated conda environment:
pip install streamlit

# Step 4: Deactivate when finished
conda deactivate""",
        language="bash",
    )

with col_conda2:
    st.markdown("#### ⚖️ Conda vs Pip: When to Use What?")
    st.markdown(
        """
- **Use Conda** when:
  - You are working in corporate environments that mandate Anaconda/Miniconda.
  - You require complex geospatial or deep learning packages (e.g. `gdal`, `geopandas`, `cudatoolkit`) with precompiled C binaries.
- **Use Pip + venv** when:
  - You want lightweight, fast setup without the overhead of Anaconda.
  - You deploy to Docker containers, AWS, GCP, or Streamlit Community Cloud.
"""
    )

st.divider()

# ==============================================================================
# 5. Verifying the Installation
# ==============================================================================
st.header("5. Verifying the Installation", divider="rainbow")

st.markdown(
    """
After installing Streamlit, verify that the CLI binary is registered and functional using these standard checks:
"""
)

col_ver1, col_ver2 = st.columns(2)

with col_ver1:
    st.markdown("#### 1. Official Streamlit Hello Demo")
    st.code(
        """# Launches the official Streamlit interactive demo in your browser
streamlit hello""",
        language="bash",
    )
    st.markdown(
        """
This starts a local development server at `http://localhost:8501` showcasing animation, DataFrame plotting, and fractal demos.
"""
    )

with col_ver2:
    st.markdown("#### 2. Programmatic Python Verification")
    st.code(
        """# One-liner to test python import and print version
python -c "import streamlit as st; print(st.__version__)" """,
        language="bash",
    )
    # Live programmatic verification
    st.success(f"✅ Verified: Streamlit **{st.__version__}** is currently active and running this application!")

st.divider()

# ==============================================================================
# 6. Checking the Streamlit Version
# ==============================================================================
st.header("6. Checking the Streamlit Version", divider="rainbow")

col_v1, col_v2 = st.columns(2)

with col_v1:
    st.markdown("#### 💻 Terminal Commands")
    st.code(
        """# 1. Direct version flag
streamlit --version
# (or shorthand)
streamlit version

# 2. Detailed package metadata via pip
pip show streamlit""",
        language="bash",
    )

with col_v2:
    st.markdown("#### 📋 Understanding `pip show streamlit` Output")
    st.markdown(
        f"""
Running `pip show streamlit` displays detailed package metadata:
- **Name**: `streamlit`
- **Version**: `{st.__version__}`
- **Summary**: `A faster way to build and share data apps.`
- **License**: `Apache-2.0`
- **Location**: `{sys.prefix}\\Lib\\site-packages`
"""
    )

st.divider()

# ==============================================================================
# 7. Understanding Streamlit Dependencies
# ==============================================================================
st.header("7. Understanding Streamlit's Internal Dependencies", divider="rainbow")

st.markdown(
    """
Streamlit is not a standalone executable; it is a Python library built upon a robust stack of open-source libraries.
Understanding these dependencies explains how Streamlit manages WebSockets, serialization, and reactivity.
"""
)

deps_df = pd.DataFrame(
    {
        "Dependency": [
            "Tornado",
            "PyArrow",
            "Altair",
            "Click",
            "Watchdog / Watchfiles",
            "Pydantic",
            "Requests",
            "Packaging",
        ],
        "Category": [
            "Web Server / WebSockets",
            "Data Serialization",
            "Data Visualization",
            "CLI Parser",
            "File System Watcher",
            "Data Validation",
            "Networking",
            "Version Utilities",
        ],
        "Role in Streamlit Architecture": [
            "Powers the async Python web server and manages bi-directional WebSocket connections between the browser and Python runtime.",
            "Columnar memory format used to serialize Pandas and Polars DataFrames efficiently to the frontend browser over WebSockets.",
            "Declarative grammar of graphics engine powering Streamlit's native statistical charts.",
            "Command-line parser that interprets `streamlit run`, flags (`--server.port`, `--server.headless`), and CLI arguments.",
            "Monitors source code files on disk for modifications to trigger automatic instant script re-execution (`runOnSave`).",
            "Validates configuration schemas and enforces type safety across internal component protocols.",
            "Handles HTTP communication, telemetry pings, and cloud deployment authentication checks.",
            "Validates dependency version compatibility requirements across runtime environments.",
        ],
    }
)

st.dataframe(deps_df, width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# 8. Creating a Production Project Folder
# ==============================================================================
st.header("8. Creating a Production Project Folder Structure", divider="rainbow")

st.markdown(
    """
When building real-world Streamlit apps for data science, ML inference, or GenAI, follow this standard directory structure:
"""
)

col_tree, col_setup = st.columns([3, 2])

with col_tree:
    st.markdown("#### 📁 Standard Project Structure")
    project_tree = """my_streamlit_app/
│── .streamlit/             # App-level configuration
│   ├── config.toml         # Port, theme (dark/light), runOnSave
│   └── secrets.toml        # Private API keys (git-ignored)
│── .venv/                  # Isolated virtual environment
│── data/                   # Data files and sample assets
│   └── sales_data.csv
│── models/                 # Serialized ML model weights
│   └── pipeline.joblib
│── pages/                  # Native multi-page application routes
│   ├── 1_Executive_Summary.py
│   └── 2_Model_Inference.py
│── app.py                  # Main landing / entrypoint script
│── requirements.txt        # Frozen package dependencies
│── .gitignore              # Files excluded from version control
└── README.md               # Setup and execution instructions"""
    st.code(project_tree, language="text")

with col_setup:
    st.markdown("#### 🛠️ Step-by-Step Setup Commands")
    st.code(
        """# 1. Create project folder and enter it
mkdir my_streamlit_app
cd my_streamlit_app

# 2. Create and activate virtual environment
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1  # (Windows)
# source .venv/bin/activate       # (Mac/Linux)

# 3. Install core libraries
pip install streamlit pandas numpy scikit-learn

# 4. Freeze dependencies into requirements.txt
pip freeze > requirements.txt

# 5. Create entrypoint file
New-Item app.py -ItemType File

# 6. Launch your application
streamlit run app.py""",
        language="powershell",
    )

st.divider()

# ==============================================================================
# Summary & Interview Checklist
# ==============================================================================
st.header("🎯 Summary: Setup & Installation Interview Checklist", divider="rainbow")

st.markdown(
    """
1. **Never install globally**: Always use an isolated virtual environment (`venv` or `conda`).
2. **Launch with `streamlit run`**: Never execute `python app.py` directly; `streamlit run` initializes the Tornado web server and WebSocket runtime.
3. **Always freeze dependencies**: Run `pip freeze > requirements.txt` so production servers and cloud platforms install identical package versions.
4. **Isolate secrets**: Keep sensitive credentials (OpenAI keys, database passwords) inside `.streamlit/secrets.toml` and never commit it to Git.
"""
)
