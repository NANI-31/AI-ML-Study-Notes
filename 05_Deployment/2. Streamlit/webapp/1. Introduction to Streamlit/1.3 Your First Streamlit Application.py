import streamlit as st

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="1.3 Your First Streamlit App | Complete Guide",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 1.3 Your First Streamlit Application — Development Workflow & Lifecycle")
st.markdown(
    """
Welcome to **Lesson 1.3: Your First Streamlit Application**.
This module walks you through the complete lifecycle of developing and running a Streamlit app:
from creating the file, understanding `import streamlit as st`, writing minimal interactive code,
running via `streamlit run`, dissecting the local Tornado server, mastering automatic reruns,
and gracefully stopping the process.
"""
)

st.divider()

# ==============================================================================
# 1. Creating a Python File
# ==============================================================================
st.header("1. Creating a Python File", divider="rainbow")

col_f1, col_f2 = st.columns(2)

with col_f1:
    st.markdown(
        """
### Best Practices for File Naming
- **Standard Convention**: Name your main entrypoint script **`app.py`** or **`main.py`**.
- **Multi-Page App Convention**: Use numbered prefixes with underscores (e.g. `1_Home.py`, `2_Dashboard.py`) inside a `pages/` directory.
- ⚠️ **Naming Rules**:
  - Do **NOT** name your file `streamlit.py`! This causes a circular import error (`AttributeError: module 'streamlit' has no attribute 'write'`).
  - Avoid spaces or special characters in script filenames when deploying to cloud environments.
"""
    )

with col_f2:
    st.markdown("#### 💻 Terminal Commands to Create `app.py`")
    st.code(
        """# Windows PowerShell:
New-Item app.py -ItemType File

# Windows Command Prompt (CMD):
type nul > app.py

# macOS & Linux (Bash / Zsh):
touch app.py""",
        language="bash",
    )

st.divider()

# ==============================================================================
# 2. Importing Streamlit
# ==============================================================================
st.header("2. Importing Streamlit (`import streamlit as st`)", divider="rainbow")

col_imp1, col_imp2 = st.columns(2)

with col_imp1:
    st.markdown("#### 📜 The Universal Convention")
    st.code(
        """import streamlit as st""",
        language="python",
    )
    st.markdown(
        """
- Across the global Python and Data Science ecosystem, Streamlit is **always** imported with the alias **`st`**.
- This matches industry conventions like `import numpy as np` and `import pandas as pd`.
"""
    )

with col_imp2:
    st.markdown("#### 🧠 What is `st` Under the Hood?")
    st.markdown(
        """
- `st` represents a singleton module that holds the **active execution context** for the current browser session.
- Every time you call a method like `st.title()` or `st.slider()`, `st` registers a widget or visual block with Streamlit's root **`DeltaGenerator`**, which queues and streams the element to the frontend.
"""
    )

st.divider()

# ==============================================================================
# 3. Writing a Basic Application
# ==============================================================================
st.header("3. Writing a Basic Application (Minimal Working Example)", divider="rainbow")

st.markdown(
    """
A complete, interactive Streamlit application requires only a few lines of readable Python code.
Compare the code on the left with the **live interactive output** on the right:
"""
)

col_code, col_live = st.columns(2)

with col_code:
    st.markdown("#### 💻 Source Code (`app.py`)")
    basic_app_code = '''import streamlit as st

st.title("My First Streamlit App 🎈")
st.write("Welcome! This is an interactive data application built in pure Python.")

# Interactive text input
user_name = st.text_input("Enter your name:", value="Data Scientist")

# Interactive slider widget
experience = st.slider("Years of Python Experience:", 0, 20, 3)

# Button trigger with response banner
if st.button("Submit Profile"):
    st.success(f"Hello {user_name}! You have {experience} years of Python experience.")'''
    st.code(basic_app_code, language="python")

with col_live:
    st.markdown("#### 🖥️ Live Embedded Output Demonstration")
    with st.container(border=True):
        st.subheader("My First Streamlit App 🎈")
        st.write("Welcome! This is an interactive data application built in pure Python.")
        demo_name = st.text_input("Enter your name:", value="Data Scientist", key="first_app_demo_name")
        demo_exp = st.slider("Years of Python Experience:", 0, 20, 3, key="first_app_demo_exp")
        if st.button("Submit Profile", key="first_app_demo_btn"):
            st.success(f"Hello {demo_name}! You have {demo_exp} years of Python experience.")

st.divider()

# ==============================================================================
# 4. Running an Application Using streamlit run
# ==============================================================================
st.header("4. Running an Application Using `streamlit run`", divider="rainbow")

col_run1, col_run2 = st.columns(2)

with col_run1:
    st.markdown("#### 💻 The Command")
    st.code(
        """# Navigate to the folder containing your script:
cd "d:\\Data\\GitHub\\my_project"

# Launch your application:
streamlit run app.py""",
        language="powershell",
    )
    st.markdown(
        """
#### ⚠️ Critical Technical Interview Question:
**Q: Why must you use `streamlit run app.py` instead of `python app.py`?**
- Running `python app.py` executes the script as a regular standalone Python script. The Streamlit functions execute, but **no web server starts**, no WebSockets are opened, and no web page is rendered.
- `streamlit run app.py` initializes the **Tornado web server**, compiles the React frontend assets, establishes real-time WebSocket communication, and launches your browser.
"""
    )

with col_run2:
    st.markdown("#### ⚙️ Useful Command-Line Flags")
    st.markdown(
        """
| Flag | Example Usage | Purpose |
| :--- | :--- | :--- |
| **`--server.port`** | `streamlit run app.py --server.port 8502` | Runs on a custom port when 8501 is occupied. |
| **`--server.headless`** | `streamlit run app.py --server.headless true` | Prevents auto-launching a browser (ideal for Docker / remote servers). |
| **`--server.runOnSave`** | `streamlit run app.py --server.runOnSave true` | Automatically reloads the web page on file save. |
| **`--browser.serverAddress`**| `streamlit run app.py --browser.serverAddress 0.0.0.0` | Binds to all network interfaces. |
| **Pass Custom Args** | `streamlit run app.py -- --env prod` | Passes CLI arguments directly to your Python script via `sys.argv`. |
"""
    )

st.divider()

# ==============================================================================
# 5. Understanding the Local Server
# ==============================================================================
st.header("5. Understanding the Local Server (Tornado & WebSockets)", divider="rainbow")

st.markdown(
    """
When you run `streamlit run app.py`, Streamlit prints two URLs to your terminal:
"""
)

st.code(
    """  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.10:8501""",
    language="text",
)

col_srv1, col_srv2 = st.columns(2)

with col_srv1:
    st.markdown(
        """
#### 🏠 Local URL vs 🌐 Network URL
- **Local URL (`http://localhost:8501` or `http://127.0.0.1:8501`)**:
  - Accessible **only from the machine running the code**.
  - Used for local development and private debugging.
- **Network URL (`http://192.168.x.x:8501`)**:
  - Uses your machine's private LAN IP address.
  - Allows **any other device** (colleague's laptop, tablet, phone) connected to the same local Wi-Fi network to test your app without cloud deployment!
"""
    )

with col_srv2:
    st.markdown(
        """
#### ⚡ How the Server Works Internally
1. **Tornado Server**: Streamlit spawns an asynchronous Python web server powered by Tornado.
2. **WebSocket Bridge**: The browser opens a persistent bi-directional WebSocket connection (`/ws`) to Tornado.
3. **Protobuf Messages**: UI elements are serialized into Google Protocol Buffers (Protobuf) and streamed to the React frontend.
4. **Zero Page Refreshes**: Only updated DOM elements are repainted—the browser never performs a full HTTP page reload!
"""
    )

st.divider()

# ==============================================================================
# 6. Opening the App in a Browser
# ==============================================================================
st.header("6. Opening the App in a Browser", divider="rainbow")

col_b1, col_b2 = st.columns(2)

with col_b1:
    st.markdown(
        """
### Automatic Browser Launching
- By default, Streamlit automatically opens your operating system's default web browser (Chrome, Edge, Firefox, Safari) immediately upon server startup.
- If your browser fails to open automatically:
  1. Open any modern browser.
  2. Navigate to [http://localhost:8501](http://localhost:8501).
"""
    )

with col_b2:
    st.markdown(
        """
### Headless / Cloud Environments
- When deploying on remote Linux VMs, Docker containers, or EC2 instances without a desktop display:
  ```bash
  streamlit run app.py --server.headless true
  ```
- This prevents Streamlit from attempting to launch a nonexistent desktop graphical browser.
"""
    )

st.divider()

# ==============================================================================
# 7. Understanding Automatic Reruns (The Reactive Mental Model)
# ==============================================================================
st.header("7. Understanding Automatic Reruns & Hot-Reloading", divider="rainbow")

st.markdown(
    """
Yes, Streamlit supports **automatic reloading on save** so the webpage updates instantly without manually refreshing.
"""
)

col_re1, col_re2 = st.columns(2)

with col_re1:
    st.markdown(
        """
### 1. In Your Currently Running Browser Tab (Instant)
While your app is open at `http://localhost:8501`:

1. Make a small edit in `2.1 Displaying Text.py` and press **`Ctrl + S`** to save.
2. In the top-right corner of the Streamlit webpage, a prompt will appear:
   > **Source file changed**  
   > `[Rerun]` &nbsp; `[Always rerun]`
3. Click **Always rerun** (or press the **`A`** key on your keyboard).
4. Alternatively, click the three dots menu (**`⋮`**) in the top-right corner ➔ **Settings** ➔ check the **"Run on save"** checkbox.
"""
    )

with col_re2:
    st.markdown(
        r"""
### 2. Permanent Global Auto-Update (Configured for You)
To make sure you never have to click "Always rerun" again, the following permanent settings have been configured:

#### ⚙️ Global Streamlit Config
Created `C:\Users\Chowd\.streamlit\config.toml` with:
```toml
[server]
runOnSave = true
```

#### 💻 PowerShell `psr` Function
Updated `psr` in `PythonTools.psm1` to include the `--server.runOnSave true` flag:
```powershell
streamlit run $filename --server.runOnSave true
```

From now on, whenever you save any file with **`Ctrl + S`**, Streamlit will detect the change and re-render the page automatically.
"""
    )

st.markdown("#### 🧪 Interactive Rerun Counter Experiment")
if "rerun_counter" not in st.session_state:
    st.session_state.rerun_counter = 1

col_cnt1, col_cnt2 = st.columns([1, 3])
with col_cnt1:
    st.info(f"🔢 Execution Count: **{st.session_state.rerun_counter}**")
with col_cnt2:
    if st.button("Trigger a Script Rerun 🔄", key="sec7_rerun_btn"):
        st.session_state.rerun_counter += 1
        st.rerun()
    st.caption("Clicking the button above invokes `st.rerun()`, executing the script from top to bottom and updating the counter.")

st.divider()

# ==============================================================================
# 8. Stopping the Application
# ==============================================================================
st.header("8. Stopping the Application", divider="rainbow")

col_stop1, col_stop2 = st.columns(2)

with col_stop1:
    st.markdown("#### 🛑 Graceful Shutdown via Keyboard")
    st.markdown(
        """
To stop the running Streamlit development server:
1. Switch to the terminal window running `streamlit run`.
2. Press **`Ctrl + C`** (or `Ctrl + Break` on Windows).
3. The server immediately closes active WebSockets and exits:
   ```text
   Stopping...
   Shutting down
   Waiting for application shutdown.
   Application shutdown complete.
   ```
"""
    )

with col_stop2:
    st.markdown("#### ⚠️ Troubleshooting a Stuck Port (Port 8501 in Use)")
    st.markdown(
        """
If you see `Port 8501 is already in use`, another background Streamlit process is still running.

**Kill stuck process on port 8501:**
```powershell
# Windows PowerShell (find and kill):
Get-Process -Id (Get-NetTCPConnection -LocalPort 8501).OwningProcess | Stop-Process -Force

# macOS / Linux:
kill -9 $(lsof -t -i:8501)
```

Or simply launch on an alternate port:
```bash
streamlit run app.py --server.port 8502
```
"""
    )

st.divider()

# ==============================================================================
# Summary & Interview Checklist
# ==============================================================================
st.header("🎯 Summary: Key Takeaways for Technical Interviews", divider="rainbow")

st.markdown(
    """
1. **Import Convention**: Always use `import streamlit as st`.
2. **Execution Command**: Always use `streamlit run app.py` (not `python app.py`) to initialize Tornado web server and WebSockets.
3. **Reactive Mental Model**: Every widget interaction and file save triggers a clean top-to-bottom re-execution of the script.
4. **Dual URLs**: `Local URL` for private local testing; `Network URL` for testing on other devices on the same Wi-Fi network.
5. **Port Management**: Default port is `8501`. Override with `--server.port` if occupied.
"""
)
