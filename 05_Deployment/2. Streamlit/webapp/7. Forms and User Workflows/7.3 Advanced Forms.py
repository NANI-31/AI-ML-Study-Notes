import streamlit as st
import pandas as pd
import re

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Advanced Forms Tutorial",
    page_icon="📑",
    layout="wide",
)

st.title("📑 Advanced Forms & Workflow Patterns — Complete Guide")
st.markdown(
    """
Real-world enterprise applications require diverse form architectures. 
This master guide implements end-to-end patterns for **developer registration**, **secure authentication**, 
**multi-parameter search engines**, **human feedback collection (RLHF)**, **structured data entry**, 
**wizard workflows**, **conditional fields**, and **schema-driven dynamic form generation**.
"""
)

st.divider()

# ==============================================================================
# 1. Registration Forms
# ==============================================================================
st.header("1. Developer & Organization Registration Form", divider="rainbow")
st.markdown(
    """
A production registration form validates identity, enforces password confirmation matching, 
and checks organizational tier selection before provisioning credentials.
"""
)

col_reg1, col_reg2 = st.columns(2)

with col_reg1:
    st.markdown("**💻 Python Source Code (Registration Architecture)**")
    code_reg = '''with st.form("developer_registration_form"):
    st.markdown("##### 🚀 Create Developer Account")
    name = st.text_input("Full Name *")
    email = st.text_input("Work Email *")
    c1, c2 = st.columns(2)
    with c1:
        pwd = st.text_input("Password *", type="password")
    with c2:
        confirm_pwd = st.text_input("Confirm Password *", type="password")
    tier = st.selectbox("API Tier", ["Starter (Free)", "Team ($99/mo)", "Enterprise"])
    terms = st.checkbox("I accept the Terms of Service & Privacy Policy *")
    
    register_btn = st.form_submit_button("Register Account", type="primary")

if register_btn:
    if not (name and email and pwd and confirm_pwd):
        st.error("Please fill in all mandatory fields.")
    elif pwd != confirm_pwd:
        st.error("Passwords do not match!")
    elif not terms:
        st.error("You must accept the terms.")
    else:
        st.success(f"Account created for {name} ({tier})!")'''
    st.code(code_reg, language="python")

with col_reg2:
    st.markdown("**🖥️ Live Interactive Registration Demo**")
    with st.form("demo_developer_registration_form"):
        st.markdown("##### 🚀 Create Developer Account")
        reg_name = st.text_input("Full Name *", placeholder="Ada Lovelace", key="reg_inp_name")
        reg_email = st.text_input("Work Email *", placeholder="ada@analytical-engine.ai", key="reg_inp_email")
        reg_c1, reg_c2 = st.columns(2)
        with reg_c1:
            reg_pwd = st.text_input("Password *", type="password", key="reg_inp_pwd")
        with reg_c2:
            reg_cpwd = st.text_input("Confirm Password *", type="password", key="reg_inp_cpwd")
        reg_tier = st.selectbox("API Tier", ["Starter (Free)", "Team ($99/mo)", "Enterprise"], key="reg_inp_tier")
        reg_terms = st.checkbox("I accept the Terms of Service & Privacy Policy *", key="reg_inp_terms")
        reg_sub = st.form_submit_button("Register Account", type="primary")

    if reg_sub:
        if not (reg_name.strip() and reg_email.strip() and reg_pwd and reg_cpwd):
            st.error("❌ Please fill in all mandatory fields marked with an asterisk (*).")
        elif reg_pwd != reg_cpwd:
            st.error("❌ Passwords do not match! Please re-type your password confirmation.")
        elif not reg_terms:
            st.error("❌ You must accept the terms of service before registering.")
        else:
            st.success(f"✅ Developer account created for **{reg_name}** under the **{reg_tier}** tier!")

st.divider()

# ==============================================================================
# 2. Login Form Interfaces
# ==============================================================================
st.header("2. Secure Authentication Login Form", divider="rainbow")
st.markdown(
    """
Authentication forms mask passwords, manage session state tokens, 
and prevent intermediate keystroke reruns.
"""
)

col_log1, col_log2 = st.columns(2)

with col_log1:
    st.markdown("**💻 Python Source Code (Login Architecture)**")
    code_login = '''with st.form("auth_login_form"):
    st.markdown("##### 🔐 Platform Authentication")
    user = st.text_input("Email / Username")
    secret = st.text_input("Password", type="password")
    remember = st.checkbox("Remember device for 30 days")
    
    login_btn = st.form_submit_button("Sign In", type="primary")

if login_btn:
    if user == "admin@ai.org" and secret == "supersecret":
        st.session_state["token"] = "jwt_auth_pass"
        st.success("Authenticated!")
    else:
        st.error("Invalid credentials.")'''
    st.code(code_login, language="python")

with col_log2:
    st.markdown("**🖥️ Live Interactive Login Demo**")
    with st.form("demo_auth_login_form"):
        st.markdown("##### 🔐 Platform Authentication")
        login_u = st.text_input("Email or Username", placeholder="admin@ai.org", key="demo_auth_u")
        login_p = st.text_input("Password", placeholder="supersecret", type="password", key="demo_auth_p")
        login_rem = st.checkbox("Remember device for 30 days", key="demo_auth_rem")
        login_sub = st.form_submit_button("Sign In", type="primary")

    if login_sub:
        if login_u == "admin@ai.org" and login_p == "supersecret":
            st.success("✅ Access Granted: Validated enterprise administrator token.")
        else:
            st.error("❌ Invalid credentials. (Hint: username is `admin@ai.org`, password is `supersecret`).")

st.divider()

# ==============================================================================
# 3. Search Forms
# ==============================================================================
st.header("3. Multi-Faceted Model & Vector Search Form", divider="rainbow")
st.markdown(
    """
Search forms allow users to set keywords, categories, and numeric bounds without triggering 
heavy database queries on every keystroke. The search only fires upon submit.
"""
)

# Mock model registry
registry_data = pd.DataFrame({
    "Model": ["Claude-3.5-Sonnet", "GPT-4o", "DeepSeek-V3", "Llama-3.3-70B", "Mistral-7B", "Gemma-2-9B"],
    "Type": ["Proprietary", "Proprietary", "Open-Weights", "Open-Weights", "Open-Weights", "Open-Weights"],
    "Context (k)": [200, 128, 128, 128, 32, 8],
    "Price / 1M ($)": [3.00, 2.50, 0.27, 0.70, 0.15, 0.25]
})

col_srch1, col_srch2 = st.columns(2)

with col_srch1:
    st.markdown("**💻 Python Source Code (Multi-Criteria Search)**")
    code_search = '''with st.form("model_search_form"):
    query = st.text_input("Search Keyword", placeholder="e.g. Llama")
    col1, col2 = st.columns(2)
    with col1:
        m_type = st.multiselect("Model Type", ["Open-Weights", "Proprietary"])
    with col2:
        max_cost = st.slider("Max Cost ($/1M)", 0.0, 5.0, 3.0)
    
    search_sub = st.form_submit_button("🔍 Execute Search")

if search_sub:
    # Query database / DataFrame atomically:
    results = df[df["Price / 1M ($)"] <= max_cost]
    st.dataframe(results)'''
    st.code(code_search, language="python")

with col_srch2:
    st.markdown("**🖥️ Live Interactive Search Engine Demo**")
    with st.form("demo_model_search_form"):
        srch_query = st.text_input("Search Keyword", placeholder="e.g. Llama, Claude, DeepSeek", key="srch_query_inp")
        s_c1, s_c2 = st.columns(2)
        with s_c1:
            srch_types = st.multiselect("Model Type", ["Open-Weights", "Proprietary"], default=["Open-Weights", "Proprietary"], key="srch_type_inp")
        with s_c2:
            srch_max_cost = st.slider("Max Budget ($/1M)", 0.10, 5.00, 3.00, step=0.10, key="srch_cost_inp")
        srch_sub = st.form_submit_button("🔍 Execute Model Search", type="primary")

    if srch_sub:
        mask = (registry_data["Type"].isin(srch_types)) & (registry_data["Price / 1M ($)"] <= srch_max_cost)
        if srch_query.strip():
            mask = mask & (registry_data["Model"].str.contains(srch_query, case=False))
        matched = registry_data[mask]
        st.caption(f"Search Results: **{len(matched)}** models found.")
        st.dataframe(matched, width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# 4. Feedback Forms (RLHF & User Sentiment)
# ==============================================================================
st.header("4. Human Feedback & RLHF Scoring Form", divider="rainbow")
st.markdown(
    """
Feedback forms capture ratings and qualitative critique:
- Setting `clear_on_submit=True` automatically resets the form after submission, preparing it for the next evaluation turn.
"""
)

if "stored_feedback_logs" not in st.session_state:
    st.session_state["stored_feedback_logs"] = []

col_fb1, col_fb2 = st.columns(2)

with col_fb1:
    st.markdown("**💻 Python Source Code (Feedback Form with auto-clear)**")
    code_feedback_form = '''with st.form("rlhf_feedback_form", clear_on_submit=True):
    st.markdown("##### ⭐️ Evaluate Assistant Response")
    rating = st.radio("Quality Rating", ["1 - Poor", "3 - Acceptable", "5 - Excellent"], horizontal=True)
    category = st.selectbox("Issue Category", ["None (Accurate)", "Hallucination", "Tone", "Formatting"])
    comments = st.text_area("Detailed Feedback")
    
    send_fb = st.form_submit_button("Submit Rating", type="primary")

if send_fb:
    st.session_state["feedback"].append({"rating": rating, "category": category})
    st.success("Feedback recorded! Form cleared.")'''
    st.code(code_feedback_form, language="python")

with col_fb2:
    st.markdown("**🖥️ Live Interactive Feedback Form Demo**")
    with st.form("demo_rlhf_feedback_form", clear_on_submit=True):
        st.markdown("##### ⭐️ Evaluate Assistant Response")
        fb_rating = st.radio("Quality Rating", ["1 - Poor", "2 - Subpar", "3 - Acceptable", "4 - Good", "5 - Superb"], index=4, horizontal=True, key="fb_inp_rate")
        fb_category = st.selectbox("Feedback Category", ["None (Accurate)", "Hallucination", "Instruction Missed", "Tone/Style"], key="fb_inp_cat")
        fb_notes = st.text_area("Qualitative Critique", placeholder="Explain why the response was helpful or inaccurate...", key="fb_inp_notes", height=70)
        fb_sub = st.form_submit_button("Submit Feedback", type="primary")

    if fb_sub:
        st.session_state["stored_feedback_logs"].append({
            "Rating": fb_rating,
            "Category": fb_category,
            "Critique": fb_notes if fb_notes.strip() else "(No text)"
        })
        st.toast("RLHF feedback recorded!", icon="✅")
        st.success("✅ Feedback successfully logged into training database.")

    if st.session_state["stored_feedback_logs"]:
        st.caption(f"Logged Evaluations in Session: **{len(st.session_state['stored_feedback_logs'])}**")
        st.dataframe(pd.DataFrame(st.session_state["stored_feedback_logs"]), width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# 5. Data Entry Forms (Direct Ingestion)
# ==============================================================================
st.header("5. Structured Data Entry Form", divider="rainbow")
st.markdown(
    """
Data entry forms collect structured tabular records and append them to in-memory DataFrames or backend databases:
"""
)

if "in_memory_catalog" not in st.session_state:
    st.session_state["in_memory_catalog"] = pd.DataFrame({
        "Model Name": ["Llama-3.1-8B", "Qwen-2.5-14B"],
        "Params (B)": [8, 14],
        "MMLU (%)": [68.4, 79.5],
        "License": ["Open-Source", "Apache-2.0"]
    })

col_de1, col_de2 = st.columns(2)

with col_de1:
    st.markdown("**💻 Python Source Code (Data Entry Architecture)**")
    code_entry = '''with st.form("catalog_entry_form", clear_on_submit=True):
    name = st.text_input("Model Name *")
    params = st.number_input("Parameter Count (B)", 1, 2000, 7)
    mmlu = st.slider("MMLU Benchmark (%)", 0.0, 100.0, 75.0)
    license = st.selectbox("License", ["Apache-2.0", "MIT", "Proprietary"])
    
    save_btn = st.form_submit_button("Append to Catalog", type="primary")

if save_btn and name.strip():
    new_row = pd.DataFrame([{"Model Name": name, "Params (B)": params, "MMLU (%)": mmlu, "License": license}])
    st.session_state["catalog"] = pd.concat([st.session_state["catalog"], new_row], ignore_index=True)'''
    st.code(code_entry, language="python")

with col_de2:
    st.markdown("**🖥️ Live Interactive Data Entry Demo**")
    with st.form("demo_catalog_entry_form", clear_on_submit=True):
        st.markdown("##### 📥 Ingest Model to Catalog")
        de_name = st.text_input("Model Name *", placeholder="e.g. Mistral-Small-24B", key="de_inp_name")
        de_c1, de_c2 = st.columns(2)
        with de_c1:
            de_params = st.number_input("Parameters (B)", min_value=1, max_value=2000, value=24, step=1, key="de_inp_p")
        with de_c2:
            de_mmlu = st.slider("MMLU Score (%)", 0.0, 100.0, 78.2, step=0.1, key="de_inp_m")
        de_lic = st.selectbox("License", ["Apache-2.0", "MIT", "Llama Community", "Proprietary"], key="de_inp_l")
        de_sub = st.form_submit_button("Append Record to Catalog", type="primary")

    if de_sub:
        if de_name.strip():
            new_record = pd.DataFrame([{
                "Model Name": de_name.strip(),
                "Params (B)": de_params,
                "MMLU (%)": de_mmlu,
                "License": de_lic
            }])
            st.session_state["in_memory_catalog"] = pd.concat([st.session_state["in_memory_catalog"], new_record], ignore_index=True)
            st.success(f"✅ Ingested **{de_name}** into active catalog!")
        else:
            st.error("Model Name is mandatory.")

    st.dataframe(st.session_state["in_memory_catalog"], width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# 6. Multi-Step Forms & 7. Conditional Fields
# ==============================================================================
st.header("6. Form-Scoped Wizards & 7. Conditional Fields", divider="rainbow")
st.markdown(
    """
- **Conditional Fields in Forms**: Because Streamlit forms do not rerun when internal toggles change, conditional fields that depend on immediate switches should either:
  1. Have the parent condition placed **outside the form** as a reactive toggle, or
  2. Use a multi-step form where Step 1 selects the mode, and Step 2 dynamically configures the corresponding conditional fields!
"""
)

col_cnd1, col_cnd2 = st.columns(2)

with col_cnd1:
    st.markdown("**💻 Python Source Code (Reactive Conditional Form Pattern)**")
    code_conditional = '''# Parent trigger outside form for instant reactivity:
storage_backend = st.selectbox("Storage Backend", ["Local Disk", "AWS S3", "GCS"])

with st.form("storage_config_form"):
    st.markdown(f"Configuring: **{storage_backend}**")
    
    if storage_backend == "AWS S3":
        s3_bucket = st.text_input("S3 Bucket Name *")
        s3_region = st.selectbox("AWS Region", ["us-east-1", "eu-west-1"])
    elif storage_backend == "GCS":
        gcs_project = st.text_input("GCP Project ID *")
    else:
        local_path = st.text_input("Local Mount Directory", value="/mnt/checkpoints")
        
    st.form_submit_button("Save Storage Config")'''
    st.code(code_conditional, language="python")

with col_cnd2:
    st.markdown("**🖥️ Live Interactive Conditional Form Demo**")
    
    chosen_backend = st.selectbox(
        "Storage Infrastructure Provider (Reactive Selector):",
        ["Local Disk", "AWS S3 Bucket", "Google Cloud Storage"],
        key="cond_backend_sel"
    )

    with st.form("demo_storage_config_form"):
        st.markdown(f"##### 💾 Storage Spec: `{chosen_backend}`")
        if chosen_backend == "AWS S3 Bucket":
            st.text_input("S3 Bucket Name *", placeholder="my-model-checkpoints-prod", key="s3_bkt")
            st.selectbox("AWS Region", ["us-east-1 (N. Virginia)", "eu-west-1 (Ireland)", "ap-northeast-1 (Tokyo)"], key="s3_reg")
        elif chosen_backend == "Google Cloud Storage":
            st.text_input("GCP Project ID *", placeholder="ai-research-corp-2026", key="gcp_prj")
            st.text_input("GCS Bucket URI *", placeholder="gs://fleet-checkpoints", key="gcp_bkt")
        else:
            st.text_input("Local Mount Path", value="/var/data/ml_checkpoints", key="loc_path")
            
        save_store = st.form_submit_button("Save Storage Configuration", type="primary")

    if save_store:
        st.success(f"✅ Storage configuration for **{chosen_backend}** saved!")

st.divider()

# ==============================================================================
# 8. Dynamic Form Generation from Data Schemas
# ==============================================================================
st.header("8. Dynamic Form Generation from Dictionary Schemas", divider="rainbow")
st.markdown(
    """
Instead of hardcoding every widget, iterate over a schema dictionary to **programmatically generate** 
an arbitrary number of form inputs:
"""
)

# Schema dictionary defining parameters dynamically
hyperparam_schema = {
    "learning_rate": {"type": "float", "default": 0.0003, "min": 1e-5, "max": 0.1, "label": "Learning Rate"},
    "weight_decay": {"type": "float", "default": 0.01, "min": 0.0, "max": 0.5, "label": "Weight Decay"},
    "warmup_ratio": {"type": "float", "default": 0.05, "min": 0.0, "max": 0.2, "label": "Warmup Ratio"},
    "optimizer": {"type": "select", "options": ["AdamW", "Lion", "SGD", "Sophia"], "label": "Optimizer"},
    "fp16_enabled": {"type": "bool", "default": True, "label": "Enable Mixed Precision (FP16)"}
}

col_dyn1, col_dyn2 = st.columns(2)

with col_dyn1:
    st.markdown("**💻 Python Source Code (Schema-Driven Form Generation)**")
    code_dynamic_form = '''schema = {
    "lr": {"type": "float", "default": 0.0003, "min": 1e-5, "max": 0.1},
    "optimizer": {"type": "select", "options": ["AdamW", "Lion", "SGD"]},
    "mixed_precision": {"type": "bool", "default": True}
}

with st.form("dynamic_schema_form"):
    captured_values = {}
    for key, spec in schema.items():
        if spec["type"] == "float":
            captured_values[key] = st.number_input(spec["label"], min_value=spec["min"], value=spec["default"])
        elif spec["type"] == "select":
            captured_values[key] = st.selectbox(spec["label"], spec["options"])
        elif spec["type"] == "bool":
            captured_values[key] = st.checkbox(spec["label"], value=spec["default"])
            
    submit = st.form_submit_button("Generate Config")'''
    st.code(code_dynamic_form, language="python")

with col_dyn2:
    st.markdown("**🖥️ Live Schema-Generated Form Demo**")
    
    with st.form("demo_dynamic_schema_form"):
        st.markdown("##### 🧬 Auto-Generated LoRA Hyperparameter Form")
        captured_data = {}
        for p_key, spec in hyperparam_schema.items():
            if spec["type"] == "float":
                captured_data[p_key] = st.number_input(
                    label=spec["label"],
                    min_value=spec["min"],
                    max_value=spec["max"],
                    value=spec["default"],
                    format="%.5f" if p_key == "learning_rate" else "%.2f",
                    key=f"dyn_num_{p_key}"
                )
            elif spec["type"] == "select":
                captured_data[p_key] = st.selectbox(
                    label=spec["label"],
                    options=spec["options"],
                    key=f"dyn_sel_{p_key}"
                )
            elif spec["type"] == "bool":
                captured_data[p_key] = st.checkbox(
                    label=spec["label"],
                    value=spec["default"],
                    key=f"dyn_chk_{p_key}"
                )
        
        dyn_sub = st.form_submit_button("Compile Generated Config", type="primary")

    if dyn_sub:
        st.success("✅ Dynamically generated configuration compiled:")
        st.json(captured_data)

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Advanced Form Patterns Cheat Sheet", divider="rainbow")
st.markdown(
    r"""
| Form Pattern | Architectural Strategy | Key Benefit |
| :--- | :--- | :--- |
| **Registration** | Password match check & mandatory term validations | Protects account integrity before API token provisioning |
| **Login** | Masked inputs (`type="password"`) & session token storage | Prevents intermediate keystroke network chatter |
| **Search Engine** | Batched multiselects and sliders submitted atomically | Avoids re-querying large vector stores on every slider drag |
| **RLHF Feedback** | `clear_on_submit=True` | Clean, frictionless continuous feedback submission |
| **Data Entry** | Concatenating new DataFrame rows upon submit | Rapid structured dataset creation and ingestion |
| **Conditional Fields** | Put parent condition outside form; render child inputs inside | Instant UI reactivity combined with batched form submission |
| **Dynamic Generation** | Loop over parameter schema dicts | Scales to hundreds of configurable model arguments effortlessly |
"""
)
