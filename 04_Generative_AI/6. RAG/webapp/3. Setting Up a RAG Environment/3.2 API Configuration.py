import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# Page Configuration
st.set_page_config(
    page_title="3.2 API Configuration - RAG Curriculum",
    page_icon="🔑",
    layout="wide"
)

# Title Block & 5-Pillar Overview
st.title("3.2 API Configuration: Multi-Provider Credentials, Secret Hygiene, & Vaults")
st.caption("Chapter 3: Setting Up a RAG Environment | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
In enterprise generative AI engineering, API keys represent direct access to proprietary data stores and corporate 
billing accounts. An accidentally exposed key can be discovered by automated scrapers within minutes, leading to 
unauthorized compute usage, data exfiltration, and severe compliance violations.

Moving beyond insecure practices (such as hardcoding tokens in code) requires mastering **Enterprise Credential Hygiene**: 
local development via **`.env` files** with strict `.gitignore` guardrails, **scoped project keys** with hard spending caps, 
**automated pre-commit leak detection**, and production secret injection via **Cloud KMS** (AWS Secrets Manager, 
Google Cloud Secret Manager, HashiCorp Vault) or **IAM Role-Based Authentication**.

In this lesson, you will master:
1. Configuring and scoping provider API credentials (OpenAI, Anthropic, Google Gemini, Hugging Face).
2. Environment variable injection mechanisms and container runtime variable management.
3. Local `.env` secret management and automated version control leak prevention.
4. Production key rotation, least-privilege scoping, and KMS secret managers.
5. Building an enterprise secret validation engine with masking and health audits.
""")

st.divider()

# Module-Scope Helper Classes for Interactive Workbench & Production Pipeline
class SecretAuditorEngine:
    def __init__(self):
        self.provider_patterns = {
            "OpenAI": {"prefix": "sk-proj-", "min_len": 40, "desc": "Project-scoped OpenAI API key"},
            "Anthropic": {"prefix": "sk-ant-", "min_len": 40, "desc": "Claude API access token"},
            "Google Gemini": {"prefix": "AIzaSy", "min_len": 35, "desc": "Google AI Studio API key"},
            "Hugging Face": {"prefix": "hf_", "min_len": 30, "desc": "Hugging Face User Access Token"}
        }

    def mask_key(self, raw_key: str) -> str:
        if len(raw_key) < 10:
            return "********"
        return raw_key[:6] + "..." + raw_key[-4:]

    def validate_key_format(self, provider: str, key_candidate: str) -> Dict[str, Any]:
        spec = self.provider_patterns.get(provider, {})
        prefix = spec.get("prefix", "")
        min_len = spec.get("min_len", 20)

        has_valid_prefix = key_candidate.startswith(prefix)
        has_valid_length = len(key_candidate) >= min_len

        is_valid = has_valid_prefix and has_valid_length
        return {
            "provider": provider,
            "masked_preview": self.mask_key(key_candidate),
            "valid_format": is_valid,
            "prefix_match": has_valid_prefix,
            "length_valid": has_valid_length,
            "description": spec.get("desc", "")
        }

# ==============================================================================
# 1. Architecture Blueprint
# ==============================================================================
st.header("1. Architecture Blueprint: Enterprise Secret Management & Zero-Leak Credential Fabric", divider="rainbow")
st.markdown("""
A production RAG deployment decouples secret storage from application code across development and deployment environments:
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_secrets = """
    digraph SecretArchitecture {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_dev {
            label="Tier 1: Local Development Secrets Tier";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            fontsize=12;
            env_file [label="Local .env File\\n(Stored on developer workstation only)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            git_ignore [label=".gitignore Guardrail\\nStrictly excludes .env and *.pem files", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            pre_commit [label="Pre-Commit Hook (Gitleaks / TruffleHog)\\nBlocks git commit if API key pattern detected", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            env_file -> git_ignore -> pre_commit;
        }

        subgraph cluster_prod {
            label="Tier 2: Production Cloud Secret Management Tier";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            fontsize=12;
            kms [label="Cloud KMS / Secret Manager\\n(AWS Secrets Manager, HashiCorp Vault, GCP Secret Manager)", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            iam [label="IAM Role Authentication (Zero Long-Lived Keys)\\nTemporary STS assume-role credentials", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            kms -> iam;
        }

        subgraph cluster_app {
            label="Tier 3: Runtime Memory Injection Tier";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            fontsize=12;
            runtime [label="Application Process Memory\\nInjected via environment variables into process space only", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            clients [label="Multi-Provider AI Clients\\nOpenAI, Anthropic, Gemini, Hugging Face SDKs", fillcolor="#1E1B4B", fontcolor="#E0E7FF", color="#818CF8"];
            runtime -> clients;
        }

        pre_commit -> runtime [label="Local Dev Flow"];
        iam -> runtime [label="Production CI/CD Flow"];
    }
    """
    st.graphviz_chart(dot_secrets, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production enterprise AI stacks have abandoned long-lived root API keys. Organizations enforce **Project-Scoped 
API Keys** with automated monthly rotation policies, hard monthly compute spend ceilings, and IP-whitelisting. 
In containerized environments (Kubernetes, AWS ECS), secrets are injected into process memory at runtime via 
vault sidecars rather than being persisted in container images or configuration files.
""")

st.divider()

# ==============================================================================
# 2. OpenAI API Keys
# ==============================================================================
st.header("2. OpenAI API Keys: Scoped Project Keys & Rate Limits", divider="rainbow")
st.markdown("""
OpenAI provides access to GPT-4o, GPT-4o-mini, and text-embedding-3 models:
* **Project-Level Scoping**: Always generate keys scoped to a specific project (`sk-proj-...`) rather than organization-wide master keys.
* **Granular Permissions**: Restrict the key to specific model endpoints (e.g. Model Read/Write only, disabling fine-tuning or admin access).
* **Usage Limits**: Configure hard spending limits in the platform billing dashboard to protect against unbounded generation loops.
""")

st.code("""
# Standard OpenAI client initialization from environment
from openai import OpenAI

client = OpenAI() # Automatically reads OPENAI_API_KEY from environment
""", language="python")

st.divider()

# ==============================================================================
# 3. Anthropic API Keys
# ==============================================================================
st.header("3. Anthropic API Keys: Claude Model Family & Quota Configuration", divider="rainbow")
st.markdown("""
Anthropic provides access to Claude 3.5 Sonnet and Haiku:
* **Key Prefix**: Begins with `sk-ant-...`.
* **Workspace Isolation**: Allocate separate workspaces for development, staging, and production environments to isolate rate-limit tiers.
* **Prompt Caching Support**: Large RAG context prompts benefit from Anthropic's prompt caching, reducing repeated retrieval token costs by up to 90%.
""")

st.code("""
# Standard Anthropic client initialization
from anthropic import Anthropic

client = Anthropic() # Reads ANTHROPIC_API_KEY from environment
""", language="python")

st.divider()

# ==============================================================================
# 4. Google Gemini API Keys
# ==============================================================================
st.header("4. Google Gemini API Keys: Vertex AI & Google AI Studio Gateways", divider="rainbow")
st.markdown("""
Google offers two access pathways for Gemini models:
* **Google AI Studio (`AIzaSy...`)**: Streamlined developer API keys suitable for prototyping and small-to-medium applications.
* **Google Cloud Vertex AI**: Enterprise production pathway utilizing Google Cloud IAM service account tokens with VPC Service Controls, ensuring customer data does not transit the public internet.
""")

st.code("""
# Standard Google GenAI client initialization
from google import genai

client = genai.Client() # Reads GEMINI_API_KEY from environment
""", language="python")

st.divider()

# ==============================================================================
# 5. Hugging Face Configuration
# ==============================================================================
st.header("5. Hugging Face Configuration: Model Hub Tokens & Inference Endpoints", divider="rainbow")
st.markdown("""
Hugging Face tokens (`hf_...`) provide access to open-weights models and managed inference endpoints:
* **User Access Tokens**: Generate fine-grained tokens with `read` permissions for model weights (e.g. gated models like Llama 3 or Mistral).
* **Local CLI Authentication**: Log in locally using `huggingface-cli login` to cache credentials securely in the user home directory.
""")

st.code("""
# Hugging Face token configuration
# bash: huggingface-cli login
from huggingface_hub import login

# Or programmatically read HUGGINGFACEHUB_API_TOKEN
""", language="python")

st.divider()

# ==============================================================================
# 6. Environment Variables
# ==============================================================================
st.header("6. Environment Variables: Cloud Runtime Variable Injection", divider="rainbow")
st.markdown("""
Environment variables allow applications to configure credentials without baking secrets into source code:
* **Process Isolation**: Environment variables exist in process memory and are inherited only by child processes.
* **Container Injection**: In Docker, inject variables at container start (`docker run -e OPENAI_API_KEY=...`) or through Kubernetes Secret manifests.
* **Zero Logging Rule**: Ensure application logging frameworks strip environment variables from diagnostic logs to prevent credential leakage.
""")

st.divider()

# ==============================================================================
# 7. .env Files
# ==============================================================================
st.header("7. `.env` Files: Local Developer Hygiene & Version Control Guardrails", divider="rainbow")
st.markdown("""
For local development, `python-dotenv` loads environment variables from a `.env` file into process memory:
""")

col_env1, col_env2 = st.columns(2)
with col_env1:
    st.subheader("Example `.env` Template")
    st.code("""
    # .env (NEVER COMMIT TO GIT)
    OPENAI_API_KEY=sk-proj-abc123xyz...
    ANTHROPIC_API_KEY=sk-ant-def456uvw...
    GEMINI_API_KEY=AIzaSyGhi789...
    HUGGINGFACEHUB_API_TOKEN=hf_jkl012...
    RAG_ENVIRONMENT=development
    """, language="bash")

with col_env2:
    st.subheader("Mandatory `.gitignore` Rules")
    st.code("""
    # .gitignore
    .env
    .env.*
    *.pem
    *.key
    credentials.json
    """, language="gitignore")

st.divider()

# ==============================================================================
# 8. API Key Security
# ==============================================================================
st.header("8. API Key Security: Secret Rotation, KMS Vaults, & Pre-Commit Detection", divider="rainbow")
st.markdown("""
An enterprise secret security posture incorporates three operational safeguards:
""")

df_sec_layers = pd.DataFrame([
    {"Security Safeguard": "1. Pre-Commit Hooks", "Tooling": "Gitleaks, TruffleHog", "Protection": "Scans staged git commits locally and rejects commits matching API key regex signatures."},
    {"Security Safeguard": "2. Secret Rotation", "Tooling": "Automated KMS Schedules", "Protection": "Rotates API keys on a 30-day cadence, automatically updating vault stores with zero downtime."},
    {"Security Safeguard": "3. Spend Anomaly Alerts", "Tooling": "Provider Webhooks & Slack Alerts", "Protection": "Notifies engineering on-call if billing consumption increases by more than 300% in a 1-hour window."}
])
st.dataframe(df_sec_layers, use_container_width=True)

st.divider()

# ==============================================================================
# 9. Production Implementation
# ==============================================================================
st.header("9. Production Implementation: Multi-Provider Credential Validator & Health Auditor", divider="rainbow")
st.markdown("""
Below is a runnable Python production script that audits configured environment variables, masks sensitive characters, 
and verifies provider readiness:
""")

prod_cred_code = """
from typing import Dict, Any, List

class MultiProviderSecretAuditor:
    def __init__(self, mock_env: Dict[str, str]):
        self.mock_env = mock_env
        self.required_providers = {
            "OPENAI_API_KEY": {"name": "OpenAI", "prefix": "sk-proj-"},
            "ANTHROPIC_API_KEY": {"name": "Anthropic", "prefix": "sk-ant-"},
            "GEMINI_API_KEY": {"name": "Google Gemini", "prefix": "AIzaSy"}
        }

    def mask_secret(self, secret: str) -> str:
        if len(secret) < 10:
            return "********"
        return secret[:6] + "..." + secret[-4:]

    def audit_credentials(self) -> Dict[str, Any]:
        report: List[Dict[str, Any]] = []
        all_valid = True

        for env_var, spec in self.required_providers.items():
            val = self.mock_env.get(env_var, "")
            is_present = bool(val)
            has_valid_prefix = val.startswith(spec["prefix"]) if is_present else False
            is_healthy = is_present and has_valid_prefix

            if not is_healthy:
                all_valid = False

            report.append({
                "provider": spec["name"],
                "env_variable": env_var,
                "is_configured": is_present,
                "has_valid_format": has_valid_prefix,
                "masked_preview": self.mask_secret(val) if is_present else "NOT_CONFIGURED"
            })

        return {
            "all_credentials_ready": all_valid,
            "provider_audit": report
        }

if __name__ == "__main__":
    # Test with simulated environment dictionary
    test_env = {
        "OPENAI_API_KEY": "sk-proj-abcdef1234567890abcdef1234567890",
        "ANTHROPIC_API_KEY": "sk-ant-api03-abcdef1234567890abcdef",
        "GEMINI_API_KEY": "AIzaSyABCDEF1234567890ABCDEF123456"
    }
    auditor = MultiProviderSecretAuditor(test_env)
    print(auditor.audit_credentials())
"""
st.code(prod_cred_code, language="python")

st.divider()

# ==============================================================================
# 10. Interactive Workbench
# ==============================================================================
st.header("10. Interactive Workbench: Secret Masking & Key Format Validation Studio", divider="rainbow")
st.markdown("""
Test how enterprise secret auditors validate provider key prefixes and length specifications:
""")

auditor_engine = SecretAuditorEngine()

col_w1, col_w2 = st.columns(2)
with col_w1:
    test_provider = st.selectbox("Select Target Provider:", list(auditor_engine.provider_patterns.keys()))
with col_w2:
    default_placeholders = {
        "OpenAI": "sk-proj-samplekey1234567890abcdefghijklmnop",
        "Anthropic": "sk-ant-sampletoken1234567890abcdefghijkl",
        "Google Gemini": "AIzaSySampleKey1234567890abcdefghijklm",
        "Hugging Face": "hf_sampleusertoken1234567890abcdefg"
    }
    input_key = st.text_input("Enter Key Candidate to Test Format:", value=default_placeholders[test_provider])

if st.button("Validate Key Format & Masking", type="primary"):
    val_res = auditor_engine.validate_key_format(test_provider, input_key)
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.subheader("Masked Preview")
        st.code(val_res["masked_preview"], language="text")
        st.caption(f"Provider Specification: {val_res['description']}")

    with col_r2:
        st.subheader("Format Audit Status")
        if val_res["valid_format"]:
            st.success("✅ Valid Provider Key Format Detected!")
        else:
            st.error("❌ Invalid Key Format! Prefix or minimum length specification failed.")

st.divider()

# ==============================================================================
# 11. Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: Why should production cloud workloads avoid long-lived API keys in favor of IAM roles?"):
    st.markdown("""
    **Answer**: Long-lived API keys are static secrets that can be leaked, cached in logs, or shared insecurely. 
    IAM roles (such as AWS IAM or GCP Service Accounts) issue temporary, short-lived security tokens (STS) that 
    expire automatically, eliminate hardcoded credentials entirely, and integrate with cloud audit trails.
    """)

with st.expander("Q2: What is the purpose of Gitleaks in a generative AI engineering pipeline?"):
    st.markdown("""
    **Answer**: Gitleaks is an automated static analysis tool that integrates into pre-commit hooks and CI/CD pipelines 
    to detect hardcoded secrets (API keys, private tokens, passwords) using regular expressions and Shannon entropy checks 
    before code can be pushed to public or internal version control repositories.
    """)

with st.expander("Q3: How do project-scoped OpenAI keys improve security compared to legacy organization keys?"):
    st.markdown("""
    **Answer**: Project-scoped keys (`sk-proj-...`) limit the blast radius of a credential compromise. If a developer's 
    project key is leaked, an attacker only gains access to that specific project's allocated models and budget, 
    preventing unauthorized access to other corporate projects, fine-tuned model weights, or administrative settings.
    """)

with st.expander("Q4: Why must .env files never be packaged inside production Docker container images?"):
    st.markdown("""
    **Answer**: Docker image layers are immutable and can be inspected by anyone with read access to the container registry 
    via `docker history` or container inspection tools. Secrets should always be injected into container process memory 
    at runtime via environment variables or mounted vault secrets.
    """)

with st.expander("Q5: What is the function of spend anomaly alerting on provider dashboards?"):
    st.markdown("""
    **Answer**: Automated billing alerts monitor real-time API token consumption. If an infinite loop or compromised key 
    causes consumption to spike, automated alerts trigger webhooks that notify on-call engineers or revoke the key automatically.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Accidental Version Control Commit)**: Committing a `.env` file containing live credentials to a public or shared GitHub repository, allowing scrapers to exfiltrate keys within seconds.")
    st.warning("**Trap 2 (Baking API Keys into Docker Images)**: Using `COPY .env .env` or `ENV OPENAI_API_KEY=...` in a Dockerfile, permanently embedding sensitive credentials into image build layers.")
with col_trap2:
    st.error("**Trap 3 (Uncapped Spending Ceilings)**: Deploying generative AI endpoints without setting hard monthly spend caps on provider billing consoles, risking unbounded financial liability from runaway agent loops.")
    st.warning("**Trap 4 (Unsanitized Logging of Environment Dumps)**: Printing `os.environ` or error tracebacks containing active API key strings to centralized logging systems (e.g. Datadog, CloudWatch).")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the security advantage of Project-Scoped API keys?")
        st.caption("Flip for answer:")
        st.info("**Back**: Restricts the blast radius of a credential compromise strictly to an isolated project and model quota.")
    with st.container(border=True):
        st.markdown("**Front**: How should secrets be provided to production Docker containers?")
        st.caption("Flip for answer:")
        st.info("**Back**: Injected at runtime via container environment variables or mounted vault volumes, never baked into image layers.")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What does a Pre-Commit secret scanner do?")
        st.caption("Flip for answer:")
        st.info("**Back**: Runs locally before `git commit` to detect and block staged API keys and credentials from entering version control.")
    with st.container(border=True):
        st.markdown("**Front**: What is the difference between AI Studio and Vertex AI Gemini keys?")
        st.caption("Flip for answer:")
        st.info("**Back**: AI Studio uses lightweight developer keys. Vertex AI utilizes enterprise GCP IAM service account authentication within private VPCs.")

st.divider()

# ==============================================================================
# 12. Self-Assessment Checklist
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I know how to obtain and configure project-scoped keys for OpenAI, Anthropic, Gemini, and Hugging Face.", value=True)
c2 = st.checkbox("I understand why `.env` files must be excluded from version control via `.gitignore`.", value=True)
c3 = st.checkbox("I know how to install and configure pre-commit hooks to block credential leaks.", value=True)
c4 = st.checkbox("I understand the danger of baking secrets into Dockerfile image build layers.", value=True)
c5 = st.checkbox("I know how to configure hard spending limits and billing alerts on provider dashboards.", value=True)
c6 = st.checkbox("I can implement runtime secret masking and format validation in application code.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 Module 3.2 Complete! You are ready to advance to Lesson 3.3: Basic RAG Project Structure.")
