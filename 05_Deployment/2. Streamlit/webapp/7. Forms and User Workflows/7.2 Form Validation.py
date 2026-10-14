import streamlit as st
import re
from datetime import date, timedelta

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Form Validation Tutorial",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Form Validation & Error Handling — Complete Guide")
st.markdown(
    """
Form validation protects your machine learning workflows from corrupt, inconsistent, or invalid inputs. 
Because Streamlit runs in Python, you can validate inputs using both **widget-level constraints** 
and **expressive Python logic** — including regex patterns, temporal date rules, 
cross-field invariants, and clear error banners.
"""
)

st.divider()

# ==============================================================================
# 1. Required Fields & 2. Empty Input Validation
# ==============================================================================
st.header("1. Required Fields & 2. Empty Input Validation", divider="rainbow")
st.markdown(
    """
Streamlit widgets return empty strings `""` or empty collections `[]` if unpopulated:
- **Visual Cue**: Suffix required field labels with a red asterisk `*` (e.g. `label="Username *"`).
- **String Sanitation**: Always call `.strip()` to prevent submissions consisting solely of whitespace.
- **Collection Validation**: Check `len(selected_items) == 0` for multiselects.
"""
)

col_req1, col_req2 = st.columns(2)

with col_req1:
    st.markdown("**💻 Python Source Code (Checking Required Fields)**")
    code_req = '''with st.form("required_fields_form"):
    st.markdown("##### 👤 Researcher Profile")
    full_name = st.text_input("Full Name *", placeholder="Dr. Jane Doe")
    affiliations = st.multiselect("Research Lab Affiliations *", ["OpenAI", "DeepMind", "Meta AI", "CERN"])
    
    submitted = st.form_submit_button("Submit Profile")

if submitted:
    errors = []
    if not full_name.strip():
        errors.append("Full Name is required and cannot be blank.")
    if not affiliations:
        errors.append("You must select at least one lab affiliation.")
        
    if errors:
        for err in errors:
            st.error(err)
    else:
        st.success("Profile verified!")'''
    st.code(code_req, language="python")

with col_req2:
    st.markdown("**🖥️ Live Interactive Validation Demo**")
    with st.form("demo_required_fields_form"):
        st.markdown("##### 👤 Researcher Profile")
        f_name = st.text_input("Full Name *", placeholder="Dr. Jane Doe", key="req_demo_name")
        f_labs = st.multiselect("Research Lab Affiliations *", ["OpenAI", "DeepMind", "Meta AI", "CERN"], key="req_demo_labs")
        req_sub = st.form_submit_button("Submit Profile", type="primary")

    if req_sub:
        req_errors = []
        if not f_name.strip():
            req_errors.append("⚠️ Full Name is required and cannot be blank.")
        if not f_labs:
            req_errors.append("⚠️ You must select at least one lab affiliation.")

        if req_errors:
            for err in req_errors:
                st.error(err)
        else:
            st.success(f"✅ Profile verified for **{f_name}** ({', '.join(f_labs)})!")

st.divider()

# ==============================================================================
# 3. Numeric Range Validation & Tensor Core Alignment
# ==============================================================================
st.header("3. Numeric Range & Hardware Alignment Validation", divider="rainbow")
st.markdown(
    """
While `st.number_input` provides `min_value` and `max_value`, advanced ML forms often require 
**custom mathematical constraints**:
- Batch sizes must be **powers of 2** or multiples of 8 to optimize GPU Tensor Core utilization.
- Learning rates must fall strictly between $10^{-6}$ and $10^{-1}$.
- Port numbers must reside in the unprivileged user range (1024 to 65535).
"""
)

col_num1, col_num2 = st.columns(2)

with col_num1:
    st.markdown("**💻 Python Source Code (Hardware Rule Validation)**")
    code_num = '''with st.form("hyperparam_num_form"):
    st.markdown("##### ⚡ Tensor Core Hyperparameters")
    batch_size = st.number_input("Batch Size", value=18, step=1)
    lr = st.number_input("Learning Rate", value=0.0003, format="%.5f")
    
    submit_num = st.form_submit_button("Validate Hyperparameters")

if submit_num:
    errors = []
    # GPU Tensor Core rule: Multiple of 8
    if batch_size % 8 != 0:
        errors.append(f"Batch size ({batch_size}) must be a multiple of 8 for Tensor Cores!")
    if not (1e-6 <= lr <= 0.1):
        errors.append(f"Learning rate ({lr}) is outside safe bounds [1e-6, 0.1]!")
        
    if errors:
        for err in errors:
            st.error(err)
    else:
        st.success(f"Hyperparameters passed Tensor Core checks!")'''
    st.code(code_num, language="python")

with col_num2:
    st.markdown("**🖥️ Live Interactive Numeric Validation Demo**")
    with st.form("demo_hyperparam_num_form"):
        st.markdown("##### ⚡ Tensor Core Hyperparameters")
        val_bs = st.number_input("Batch Size", min_value=1, max_value=512, value=18, step=1, key="num_val_bs")
        val_lr = st.number_input("Learning Rate", value=0.0003, format="%.5f", key="num_val_lr")
        num_sub = st.form_submit_button("Validate Hyperparameters", type="primary")

    if num_sub:
        num_errors = []
        if val_bs % 8 != 0:
            num_errors.append(f"⚠️ Batch size **{val_bs}** is not a multiple of 8 (Optimal: 16, 24, 32, 64)!")
        if not (1e-6 <= val_lr <= 0.1):
            num_errors.append(f"⚠️ Learning rate **{val_lr}** is outside standard bounds [1e-6, 0.1]!")

        if num_errors:
            for err in num_errors:
                st.error(err)
        else:
            st.success(f"✅ Passed! Batch size {val_bs} aligns with GPU Tensor Cores; LR={val_lr}.")

st.divider()

# ==============================================================================
# 4. Email Format Validation & Regex Checks
# ==============================================================================
st.header("4. Email & API Key Format Regex Validation", divider="rainbow")
st.markdown(
    r"""
Use Python's standard `re` module to test string conformity:
- **Email Regex**: `r"^[\w\.-]+@[\w\.-]+\.\w+$"`
- **Hugging Face Token Regex**: `r"^hf_[A-Za-z0-9]{34,}$"`
- **OpenAI API Key Regex**: `r"^sk-(proj-)?[A-Za-z0-9_-]{32,}$"`
"""
)

col_em1, col_em2 = st.columns(2)

with col_em1:
    st.markdown("**💻 Python Source Code (Regex Validation)**")
    code_regex = '''import re

EMAIL_PATTERN = r"^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"
HF_TOKEN_PATTERN = r"^hf_[A-Za-z0-9]{10,}$"

with st.form("api_credential_form"):
    email = st.text_input("Notification Email *")
    hf_token = st.text_input("Hugging Face API Token *", type="password")
    submit_creds = st.form_submit_button("Verify Credentials")

if submit_creds:
    if not re.match(EMAIL_PATTERN, email):
        st.error("Invalid email format (e.g. user@domain.com).")
    elif not re.match(HF_TOKEN_PATTERN, hf_token):
        st.error("Invalid HF token format (must start with 'hf_').")
    else:
        st.success("Credentials validated!")'''
    st.code(code_regex, language="python")

with col_em2:
    st.markdown("**🖥️ Live Interactive Regex Validation Demo**")
    EMAIL_REGEX = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    HF_TOKEN_REGEX = r"^hf_[A-Za-z0-9]{10,}$"

    with st.form("demo_api_credential_form"):
        st.markdown("##### 🔑 Developer Credentials")
        cred_email = st.text_input("Notification Email *", placeholder="developer@lab.ai", key="cred_demo_email")
        cred_token = st.text_input("Hugging Face API Token *", placeholder="hf_...", type="password", key="cred_demo_token")
        cred_sub = st.form_submit_button("Verify Credentials", type="primary")

    if cred_sub:
        if not re.match(EMAIL_REGEX, cred_email.strip()):
            st.error("❌ Invalid email format! Please provide a valid address (e.g. `user@domain.org`).")
        elif not re.match(HF_TOKEN_REGEX, cred_token.strip()):
            st.error("❌ Invalid Hugging Face token! Must start with `hf_` followed by at least 10 characters.")
        else:
            st.success("✅ Credentials syntax verified successfully!")

st.divider()

# ==============================================================================
# 5. Date Interval Validation
# ==============================================================================
st.header("5. Date Range & Temporal Window Validation", divider="rainbow")
st.markdown(
    """
When capturing training dataset intervals or evaluation cutoffs:
- **Chronological Consistency**: Ensure `start_date <= end_date`.
- **Lookahead Guard**: Prevent `end_date > date.today()`.
- **Duration Windows**: Enforce maximum window spans (e.g. training window cannot exceed 365 days).
"""
)

col_dt1, col_dt2 = st.columns(2)

with col_dt1:
    st.markdown("**💻 Python Source Code (Temporal Window Logic)**")
    code_dates = '''with st.form("dataset_date_form"):
    st.markdown("##### 📅 Dataset Ingestion Window")
    start = st.date_input("Start Date", value=date(2026, 1, 1))
    end = st.date_input("End Date", value=date(2025, 12, 31))  # Invalid!
    
    sub_dates = st.form_submit_button("Verify Window")

if sub_dates:
    if start > end:
        st.error(f"Invalid Window: Start date ({start}) cannot be after End date ({end})!")
    elif (end - start).days > 365:
        st.error("Window exceeds maximum allowed span of 365 days.")
    else:
        st.success(f"Valid {(end - start).days}-day ingestion window!")'''
    st.code(code_dates, language="python")

with col_dt2:
    st.markdown("**🖥️ Live Interactive Date Validation Demo**")
    with st.form("demo_dataset_date_form"):
        st.markdown("##### 📅 Dataset Ingestion Window")
        c_d1, c_d2 = st.columns(2)
        with c_d1:
            val_start = st.date_input("Start Date", value=date(2026, 6, 1), key="dt_val_start")
        with c_d2:
            val_end = st.date_input("End Date", value=date(2026, 5, 1), key="dt_val_end")
        dt_sub = st.form_submit_button("Verify Window", type="primary")

    if dt_sub:
        if val_start > val_end:
            st.error(f"❌ Chronological Error: Start date (`{val_start}`) is after End date (`{val_end}`).")
        elif (val_end - val_start).days > 365:
            st.error(f"❌ Range Error: Window spans {(val_end - val_start).days} days (Maximum allowed: 365 days).")
        else:
            st.success(f"✅ Valid window: **{(val_end - val_start).days}** days configured.")

st.divider()

# ==============================================================================
# 6. Cross-Field Validation (Dependent Multi-Field Rules)
# ==============================================================================
st.header("6. Cross-Field Dependent Invariants", divider="rainbow")
st.markdown(
    """
Cross-field validation tests relationships between multiple fields:
- **Conditional Dependency**: If "Enable Custom Scheduler" is toggled ON, the warmup steps input must be $> 0$.
- **Password Confirmation**: Passwords must match exactly.
- **Budget Partitioning**: Train, Validation, and Test percentages must sum to exactly **100%**.
"""
)

col_cf1, col_cf2 = st.columns(2)

with col_cf1:
    st.markdown("**💻 Python Source Code (Cross-Field Sum Validation)**")
    code_cross = '''with st.form("split_budget_form"):
    st.markdown("##### 📊 Train / Validation / Test Ratio")
    train_pct = st.number_input("Train (%)", value=80, min_value=0, max_value=100)
    val_pct = st.number_input("Val (%)", value=15, min_value=0, max_value=100)
    test_pct = st.number_input("Test (%)", value=10, min_value=0, max_value=100)
    
    sub_split = st.form_submit_button("Validate Split Ratio")

if sub_split:
    total = train_pct + val_pct + test_pct
    if total != 100:
        st.error(f"Sum must equal 100%! Current sum is {total}%.")
    else:
        st.success("Split ratio perfectly balanced at 100%!")'''
    st.code(code_cross, language="python")

with col_cf2:
    st.markdown("**🖥️ Live Interactive Cross-Field Demo**")
    with st.form("demo_split_budget_form"):
        st.markdown("##### 📊 Train / Validation / Test Ratio")
        cf_c1, cf_c2, cf_c3 = st.columns(3)
        with cf_c1:
            tr_pct = st.number_input("Train (%)", value=70, min_value=0, max_value=100, key="cf_tr")
        with cf_c2:
            vl_pct = st.number_input("Val (%)", value=20, min_value=0, max_value=100, key="cf_vl")
        with cf_c3:
            ts_pct = st.number_input("Test (%)", value=15, min_value=0, max_value=100, key="cf_ts")
        cf_sub = st.form_submit_button("Validate Split Ratio", type="primary")

    if cf_sub:
        total_split = tr_pct + vl_pct + ts_pct
        if total_split != 100:
            st.error(f"❌ Invariant Violation: Train ({tr_pct}) + Val ({vl_pct}) + Test ({ts_pct}) = **{total_split}%** (Must sum to exactly **100%**).")
        else:
            st.success(f"✅ Invariant verified: Train ({tr_pct}%) / Val ({vl_pct}%) / Test ({ts_pct}%) = **100%**.")

st.divider()

# ==============================================================================
# 7. Error Messages & 8. Success Feedback Messages
# ==============================================================================
st.header("7. Actionable Error Banners & 8. Success Confirmations", divider="rainbow")
st.markdown(
    """
Best practices for user feedback upon submission:
1. **Accumulate All Errors**: Don't stop at the first error; collect all issues so the user can fix everything at once.
2. **Clear Feedback**: Use `st.success()`, `st.toast()`, or `st.balloons()` to clearly confirm successful operations.
"""
)

col_fb1, col_fb2 = st.columns(2)

with col_fb1:
    st.markdown("**💻 Python Source Code (Feedback Patterns)**")
    code_feedback = '''if form_submitted:
    if validation_errors:
        st.error("Please resolve the following issues:")
        for err in validation_errors:
            st.markdown(f"- {err}")
    else:
        st.success("Deployment configuration verified!")
        st.toast("Job queued successfully!", icon="🚀")'''
    st.code(code_feedback, language="python")

with col_fb2:
    st.markdown("**🖥️ Rendered Feedback Demonstration**")
    st.error("Please resolve the following issues:\n- Batch size must be a multiple of 8.\n- Notification email format is invalid.")
    st.success("✅ Training configuration verified and queued successfully!")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Form Validation Rules of Thumb", divider="rainbow")
st.markdown(
    r"""
| Validation Category | Recommended Technique | Python Verification Syntax |
| :--- | :--- | :--- |
| **Required Strings** | String stripping | `if not value.strip(): raise ...` |
| **Required Lists** | Length test | `if len(multiselect_items) == 0: raise ...` |
| **Numeric Ranges** | Bounds comparison | `if not (min_val <= val <= max_val): raise ...` |
| **Tensor Core Math** | Modulo arithmetic | `if val % 8 != 0: raise ...` |
| **Email / Tokens** | Regex matching | `if not re.match(PATTERN, val): raise ...` |
| **Date Ranges** | Chronological check | `if start_date > end_date: raise ...` |
| **Cross-Field Sums** | Aggregate checks | `if sum(percentages) != 100: raise ...` |
| **Feedback UI** | List accumulation | Collect errors in `list[str]` and render via `st.error()` |
"""
)
