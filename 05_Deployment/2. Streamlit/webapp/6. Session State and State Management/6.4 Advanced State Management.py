import streamlit as st
import pandas as pd
from datetime import datetime

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Advanced State Management Tutorial",
    page_icon="🧩",
    layout="wide",
)

st.title("🧩 Advanced State Management Patterns — Complete Guide")
st.markdown(
    """
Beyond simple counters and widget bindings, real-world enterprise applications require sophisticated state patterns. 
This master guide covers **multi-step wizard state machines**, **cross-page state persistence**, 
**shopping cart & resource allocation collections**, **LLM chat history buffers**, 
**global filter preservation**, application state resets, and architectural comparisons between session state and persistent storage.
"""
)

st.divider()

# ==============================================================================
# 1. Multi-Step Forms (Wizard State Machine)
# ==============================================================================
st.header("1. Multi-Step Wizard Forms (State Machine Pattern)", divider="rainbow")
st.markdown(
    """
A wizard guides users through complex configurations across multiple sequential steps:
- **`st.session_state["step"]`**: Tracks the active step index (`1`, `2`, `3`).
- **Data Accumulation**: Values captured in Step 1 remain safely preserved in `st.session_state` while the user completes Step 2.
- **Review & Confirm**: Step 3 presents an executive summary before final submission.
"""
)

# Initialize wizard state
if "wizard_step" not in st.session_state:
    st.session_state["wizard_step"] = 1
if "wiz_sel_model" not in st.session_state:
    st.session_state["wiz_sel_model"] = "Llama-3.3-70B-Instruct"
if "wiz_gpu_count" not in st.session_state:
    st.session_state["wiz_gpu_count"] = 4
if "wiz_radio_quant" not in st.session_state:
    st.session_state["wiz_radio_quant"] = "FP8 (Recommended)"

def next_step():
    st.session_state["wizard_step"] += 1

def prev_step():
    st.session_state["wizard_step"] -= 1

def reset_wizard():
    st.session_state["wizard_step"] = 1
    st.session_state["wiz_sel_model"] = "Llama-3.3-70B-Instruct"
    st.session_state["wiz_gpu_count"] = 4
    st.session_state["wiz_radio_quant"] = "FP8 (Recommended)"

col_wz1, col_wz2 = st.columns(2)

with col_wz1:
    st.markdown("**💻 Python Source Code (Wizard State Machine)**")
    code_wizard = '''# Wizard state machine controller:
step = st.session_state.get("step", 1)

if step == 1:
    st.subheader("Step 1: Select Model")
    st.session_state["model"] = st.selectbox("Model", ["Llama-3", "Claude", "Mistral"])
    st.button("Next ➡️", on_click=lambda: st.session_state.update(step=2))

elif step == 2:
    st.subheader("Step 2: Hardware Allocation")
    st.session_state["gpus"] = st.slider("GPU Count", 1, 8, 4)
    c1, c2 = st.columns(2)
    c1.button("⬅️ Back", on_click=lambda: st.session_state.update(step=1))
    c2.button("Next ➡️", on_click=lambda: st.session_state.update(step=3))

elif step == 3:
    st.subheader("Step 3: Review & Deploy")
    st.json({"Model": st.session_state["model"], "GPUs": st.session_state["gpus"]})
    st.button("🚀 Deploy Fleet", on_click=deploy_fleet)'''
    st.code(code_wizard, language="python")

with col_wz2:
    st.markdown("**🖥️ Live Interactive Multi-Step Deployment Wizard**")
    
    current_step = st.session_state["wizard_step"]
    st.progress(current_step / 3, text=f"Deployment Wizard: Step {current_step} of 3")
    
    with st.container(border=True):
        if current_step == 1:
            st.markdown("##### 📦 Step 1: Foundation Model Selection")
            st.selectbox(
                "Select Model Checkpoint",
                ["Llama-3.3-70B-Instruct", "DeepSeek-V3", "Mistral-Large-2"],
                key="wiz_sel_model"
            )
            st.button("Proceed to Hardware Setup ➡️", on_click=next_step, key="btn_wiz_step1")

        elif current_step == 2:
            st.markdown("##### ⚡ Step 2: Compute Hardware & Precision")
            st.slider(
                "NVIDIA H100 Replicas", 1, 16, 
                key="wiz_gpu_count"
            )
            st.radio(
                "Model Precision Format",
                ["FP8 (Recommended)", "BF16 (Native)", "INT4 (Quantized)"],
                key="wiz_radio_quant"
            )
            w_c1, w_c2 = st.columns(2)
            w_c1.button("⬅️ Back to Step 1", on_click=prev_step, key="btn_wiz_back2")
            w_c2.button("Review Configuration ➡️", on_click=next_step, key="btn_wiz_next2")

        elif current_step == 3:
            st.markdown("##### 🚀 Step 3: Review Deployment Specification")
            st.json({
                "Model Architecture": st.session_state["wiz_sel_model"],
                "GPU Replicas": f"{st.session_state['wiz_gpu_count']}x H100 (80GB)",
                "Weight Precision": st.session_state["wiz_radio_quant"],
                "Estimated Cluster Hourly Spend": f"${st.session_state['wiz_gpu_count'] * 3.85:.2f} / hr"
            })
            w_c3, w_c4 = st.columns(2)
            w_c3.button("⬅️ Back to Step 2", on_click=prev_step, key="btn_wiz_back3")
            if w_c4.button("✅ Confirm & Launch", type="primary", key="btn_wiz_confirm"):
                st.toast("Cluster deployment successfully initiated!")
                st.success("Fleet provisioned! Resetting wizard...")
                reset_wizard()
                st.rerun()

st.divider()

# ==============================================================================
# 2. Multi-Page State Persistence
# ==============================================================================
st.header("2. Multi-Page State Persistence", divider="rainbow")
st.markdown(
    """
In multi-page Streamlit apps (`st.navigation` or the `pages/` directory):
- **Universal Scope**: `st.session_state` is **shared across all pages** for that user tab.
- **Cross-Page Workflows**: A user authenticates on `Login.py`, and all subsequent analytics pages immediately read `st.session_state["user_profile"]`!
"""
)

col_mp1, col_mp2 = st.columns(2)

with col_mp1:
    st.markdown("**💻 Python Source Code (Cross-Page State Architecture)**")
    code_multipage = '''# Page 1 (Login / Configuration):
if st.button("Authenticate"):
    st.session_state["user"] = {"id": "usr_99", "role": "admin"}
    st.session_state["authenticated"] = True

# Page 2 (Analytics Dashboard):
# Reads the exact same session state!
if not st.session_state.get("authenticated", False):
    st.warning("Please authenticate on the login page first.")
    st.stop()

st.write(f"Welcome, {st.session_state['user']['id']}!")'''
    st.code(code_multipage, language="python")

with col_mp2:
    st.markdown("**🖥️ Architecture Reference Diagram**")
    st.info(
        """
""")

st.graphviz_chart(
    r'''
    digraph SessionStateArch {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        state [label="st.session_state (Client Tab Context)\n{'auth_user': 'Ada', 'tier': 'Enterprise'}", fillcolor="#DBEAFE", color="#1D4ED8"];
        p1 [label="Page 1: Login\nSets session state keys", fillcolor="#FEF3C7", color="#D97706"];
        p2 [label="Page 2: Models\nReads user parameters", fillcolor="#FEF3C7", color="#D97706"];
        p3 [label="Page 3: Billing\nEnforces subscription tier", fillcolor="#D1FAE5", color="#059669"];

        state -> p1;
        state -> p2;
        state -> p3;
    }
    '''
)

st.markdown("""
        - Data set on any page is instantly accessible to every other page.
        """
    )

st.divider()

# ==============================================================================
# 3. Shopping Cart & Resource Allocation State
# ==============================================================================
st.header("3. Collection Management: Cart & Resource Allocation", divider="rainbow")
st.markdown(
    """
Managing lists or dictionaries of items (adding, updating quantity, removing by ID):
- Store items as a dictionary or list in `st.session_state["cart"]`.
- Use helper functions to add or delete items by key.
"""
)

if "gpu_cluster_cart" not in st.session_state:
    st.session_state["gpu_cluster_cart"] = [
        {"id": "node-1", "type": "NVIDIA H100 (80GB)", "qty": 4, "price": 3.85},
        {"id": "node-2", "type": "NVIDIA A100 (80GB)", "qty": 8, "price": 2.50}
    ]

def add_gpu_node(node_type: str, qty: int, price: float):
    st.session_state["gpu_cluster_cart"].append({
        "id": f"node-{len(st.session_state['gpu_cluster_cart']) + 1}",
        "type": node_type,
        "qty": qty,
        "price": price
    })

def remove_gpu_node(index: int):
    st.session_state["gpu_cluster_cart"].pop(index)

col_crt1, col_crt2 = st.columns(2)

with col_crt1:
    st.markdown("**💻 Python Source Code (Cart Management)**")
    code_cart = '''# Add node to collection:
def add_node(gpu_type, count, rate):
    st.session_state["cart"].append({
        "type": gpu_type, "count": count, "rate": rate
    })

# Remove node by index:
def remove_node(idx):
    st.session_state["cart"].pop(idx)

# Calculate dynamic aggregate total:
hourly_spend = sum(item["count"] * item["rate"] for item in st.session_state["cart"])
st.metric("Total Hourly Burn", f"${hourly_spend:.2f} / hr")'''
    st.code(code_cart, language="python")

with col_crt2:
    st.markdown("**🖥️ Live Interactive GPU Allocation Cart**")
    
    # Calculate live totals
    total_hourly = sum(item["qty"] * item["price"] for item in st.session_state["gpu_cluster_cart"])
    total_gpus = sum(item["qty"] for item in st.session_state["gpu_cluster_cart"])
    
    st.metric("Total Fleet Hourly Spend", f"${total_hourly:.2f} / hr", f"{total_gpus} GPUs Allocated")
    
    with st.container(border=True):
        st.markdown("##### 🛒 Allocated Cluster Nodes:")
        for idx, item in enumerate(st.session_state["gpu_cluster_cart"]):
            c_desc, c_del = st.columns([4, 1], vertical_alignment="center")
            with c_desc:
                st.write(f"• **{item['qty']}x {item['type']}** @ ${item['price']:.2f}/hr (Total: ${item['qty'] * item['price']:.2f}/hr)")
            with c_del:
                st.button("❌ Remove", key=f"del_cart_{idx}_{item['id']}", on_click=remove_gpu_node, args=(idx,))

        st.divider()
        if st.button("➕ Provision Additional 2x L40S Nodes ($1.50/hr)", key="btn_add_cart_node"):
            add_gpu_node("NVIDIA L40S (48GB)", 2, 1.50)
            st.rerun()

st.divider()

# ==============================================================================
# 4. Chat History State (`st.chat_message`)
# ==============================================================================
st.header("4. Conversational Chat History State", divider="rainbow")
st.markdown(
    """
Chatbots require storing message arrays `[{"role": "user"|"assistant", "content": "..."}]` in session state:
- When a user submits a prompt, append the user message to `st.session_state["chat_history"]`.
- Simulate or stream the assistant response, then append the assistant response.
- On every rerun, loop through `st.session_state["chat_history"]` and render via `st.chat_message()`.
"""
)

if "llm_chat_history" not in st.session_state:
    st.session_state["llm_chat_history"] = [
        {"role": "assistant", "content": "Hello! I am your AI assistant. How can I help you today?"}
    ]

col_cht1, col_cht2 = st.columns(2)

with col_cht1:
    st.markdown("**💻 Python Source Code (Chat Loop Architecture)**")
    code_chat = '''# 1. Initialize message history:
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Render previous messages:
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 3. Receive new user input:
if prompt := st.chat_input("Ask a question..."):
    # Append user turn
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Generate assistant turn
    response = f"Simulated echo: {prompt}"
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()'''
    st.code(code_chat, language="python")

with col_cht2:
    st.markdown("**🖥️ Live Interactive Chat History Buffer**")
    
    with st.container(height=200, border=True):
        for msg in st.session_state["llm_chat_history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    col_inp, col_snd = st.columns([4, 1], vertical_alignment="bottom")
    with col_inp:
        chat_query = st.text_input("Send prompt:", placeholder="Ask about PyTorch, Streamlit, etc...", key="txt_chat_entry")
    with col_snd:
        if st.button("Send 💬", key="btn_send_chat"):
            if chat_query:
                st.session_state["llm_chat_history"].append({"role": "user", "content": chat_query})
                # Simulate assistant response
                st.session_state["llm_chat_history"].append({
                    "role": "assistant", 
                    "content": f"Acknowledged: '{chat_query}'. Recorded in session history."
                })
                st.rerun()

st.divider()

# ==============================================================================
# 5. Filter Persistence & 6. Resetting Application State
# ==============================================================================
st.header("5. Filter Persistence & 6. Complete State Reset Patterns", divider="rainbow")
st.markdown(
    """
- **Filter Persistence**: Storing search terms, active categories, and page offsets in `st.session_state` ensures that interacting with data tables or downloading CSVs doesn't reset filters.
- **State Reset Pattern**: A "Reset All Application State" button should cleanly re-initialize default keys without throwing `KeyError` exceptions.
"""
)

def reset_entire_application_state():
    # Retain only necessary system keys or re-initialize defaults
    keys_to_clear = [k for k in st.session_state.keys() if not k.startswith("_")]
    for k in keys_to_clear:
        del st.session_state[k]

col_rst1, col_rst2 = st.columns(2)

with col_rst1:
    st.markdown("**💻 Python Source Code (Safe Global State Reset)**")
    code_global_reset = '''def reset_app_state():
    # Safely clear all user keys:
    for key in list(st.session_state.keys()):
        if not key.startswith("_"):
            del st.session_state[key]
    st.toast("Application state purged to factory defaults!")
    st.rerun()

st.button("🔥 Factory Reset All State", on_click=reset_app_state)'''
    st.code(code_global_reset, language="python")

with col_rst2:
    st.markdown("**🖥️ Live State Inspector & Factory Reset Demo**")
    st.button("🔥 Factory Reset Session State", on_click=reset_entire_application_state, key="btn_factory_reset")
    st.caption("Active Session State Keys Count: **" + str(len([k for k in st.session_state.keys() if not k.startswith('_')])) + "** registered keys.")

st.divider()

# ==============================================================================
# 7. Per-Session User Data & 8. Session State vs. Persistent Storage
# ==============================================================================
st.header("7. Per-Session Isolation & 8. Session State vs. Persistent Storage", divider="rainbow")
st.markdown(
    r"""
Understanding where data should live is a fundamental architectural decision in production systems:
"""
)

st.markdown(
    r"""
| Storage Tier | Lifetime / Scope | Storage Location | Best Used For |
| :--- | :--- | :--- | :--- |
| **`st.session_state`** | Ephemeral; lasts only while browser tab is open | Server RAM (keyed per session UUID) | Active wizard steps, chat buffers, in-flight form inputs |
| **`st.query_params`** | Shareable via URL; survives browser refresh | Browser URL bar | Search queries, selected item IDs, permalinks |
| **Local Cookies / Web Storage**| Persistent per client device across browser restarts | Client browser | User theme preferences, remember-me auth tokens |
| **Database (PostgreSQL / SQLite)**| Permanent; shared across all users and sessions | Persistent disk / DB server | User accounts, processed invoices, production models |
| **Object Store (AWS S3 / GCS)** | Permanent durable blob storage | Cloud storage bucket | Checkpoint `.pt` weights, dataset CSVs, PDF docs |
"""
)
