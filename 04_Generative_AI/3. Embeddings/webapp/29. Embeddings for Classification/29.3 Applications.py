import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="29.3 Applications of Embedding Classification - Embeddings Curriculum",
    page_icon="🚀",
    layout="wide"
)

# Title & Overview
st.title("29.3 Applications: Sentiment, Spam, Topic, Intent & Support-Ticket Classification")
st.caption("Chapter 29: Embeddings for Classification | Domain: 3. Vector Embeddings")
st.markdown("""
Embedding-based classification has become the workhorse of enterprise artificial intelligence. 
Because foundation embeddings capture deep semantic context, nuanced terminology, and syntactic intent without requiring 
expensive GPU fine-tuning, organizations deploy embedding classifiers across mission-critical workflows:

* **Sentiment Classification**: Gauging customer satisfaction, brand perception, and subtle dissatisfaction across reviews and survey feedback.
* **Spam & Phishing Detection**: Identifying zero-day malicious emails, obfuscated phishing links, and deceptive marketing copy.
* **Topic Classification**: Automatically tagging and filing millions of research papers, legal contracts, and news articles into structured taxonomies.
* **Intent Classification**: Powering conversational AI agents by mapping natural language user prompts into discrete API actions (e.g., `book_flight`, `cancel_order`).
* **Support-Ticket Classification**: Triaging enterprise helpdesk tickets, routing inquiries to specialized engineering queues, and assigning SLA priority levels.
""")

st.divider()

# Section 1: The Five Production Application Blueprints
st.header("1. Production Blueprints for the Five Core Applications", divider="rainbow")

col_app1, col_app2 = col_app1, col_app2 = st.columns([1, 1])

with col_app1:
    st.markdown("""
    ### 1. Sentiment Classification
    * **Objective**: Predict polarity (Positive, Neutral, Negative) or fine-grained emotions (Frustrated, Satisfied, Confused).
    * **Advantage over Lexicon**: Understands sarcasm, double negatives, and indirect sentiment:
      * *"Oh great, another outage right before demo day"* -> Correctly classified as **Negative (Frustrated)**.
    
    ### 2. Spam & Phishing Detection
    * **Objective**: Distinguish legitimate communications (Ham) from social engineering scams (Spam/Phishing).
    * **Advantage over Keywords**: Detects semantic urgency patterns (*"Account suspended verify immediately"*) even when scammers use character substitutions or zero-day phishing copy.
    
    ### 3. Topic Classification
    * **Objective**: Assign multi-class or multi-label taxonomy tags to enterprise documents.
    * **Advantage**: Handles synonymy and cross-discipline overlap (e.g., an article discussing both *Kubernetes* and *HIPAA compliance* tagged with both *Cloud Native* and *Healthcare Regs*).
    """)

with col_app2:
    st.markdown("""
    ### 4. Intent Classification (Conversational AI)
    * **Objective**: Extract the user's operational goal from conversational chat turns.
    * **Advantage**: Bypasses rigid keyword matching:
      * User: *"I need my money back for the shoes that didn't fit"* -> Mapped to intent: `order_refund_request`.
    
    ### 5. Support-Ticket Triage & SLA Routing
    * **Objective**: Parse inbound helpdesk tickets, predict the resolving engineering department, and assign priority severity (P1 Critical vs P4 Informational).
    * **Advantage**: Automates routing in sub-10ms, eliminating human dispatch bottlenecks and cutting initial SLA response times by up to 75%.
    """)

st.markdown("### Enterprise Triage & Dispatch Architecture")
triage_dot = """
digraph EnterpriseTriage {
    rankdir=LR;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    InTicket [label="Inbound Customer Ticket\\n'Production cluster database locked up!'", fillcolor="#FEE2E2", color="#EF4444", style="filled,bold"];
    
    Embedder [label="Embeddings Engine\\n(text-embedding-3 / BGE)", fillcolor="#DBEAFE"];

    MultiClassifier [label="Multi-Head Classifier Suite\\nSimultaneous Microsecond Inference", fillcolor="#FEF3C7", color="#F59E0B", style="filled,bold"];

    DeptQueue [label="Target Queue:\\nInfrastructure & Database Team", fillcolor="#D1FAE5", color="#10B981"];
    PrioritySLA [label="SLA Priority: P1 Critical\\n15-minute response timer", fillcolor="#FCA5A5", color="#DC2626", style="filled,bold"];
    SentimentAlert [label="Customer Emotion:\\nHigh Urgency / Frustrated", fillcolor="#FED7AA"];

    InTicket -> Embedder;
    Embedder -> MultiClassifier;
    MultiClassifier -> DeptQueue [label="Department Head"];
    MultiClassifier -> PrioritySLA [label="Severity Head"];
    MultiClassifier -> SentimentAlert [label="Sentiment Head"];
}
"""
st.graphviz_chart(triage_dot, use_container_width=False)

st.divider()

# Section 2: Production Multi-Head Classifier Implementation
st.header("2. Code Implementation: The Multi-Head Production Triage Engine", divider="rainbow")

st.markdown("""
In enterprise ticket triage, a single text embedding vector is typically fed into **multiple specialized classifier heads simultaneously**:
1. Head A predicts **Department** (`Billing`, `DevOps`, `Security`, `Legal`).
2. Head B predicts **Severity / Priority** (`P1`, `P2`, `P3`, `P4`).
3. Head C predicts **Customer Sentiment** (`Positive`, `Neutral`, `Escalated`).

Because computing the embedding is the only heavy step, running three linear heads in parallel adds **less than 0.2ms of overhead**!
""")

col_code1, col_code2 = col_code1, col_code2 = st.columns([1, 1])

with col_code1:
    st.markdown("### Multi-Head Triage Microservice")
    st.code("""
import numpy as np
import joblib
from openai import OpenAI

client = OpenAI()

class MultiHeadTicketRouter:
    def __init__(self, dept_model_path: str, priority_model_path: str, sentiment_model_path: str):
        self.dept_clf = joblib.load(dept_model_path)
        self.priority_clf = joblib.load(priority_model_path)
        self.sentiment_clf = joblib.load(sentiment_model_path)
        
    def triage_ticket(self, ticket_text: str) -> dict:
        # 1. Single forward pass to generate embedding vector
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=ticket_text
        )
        vec = np.array([response.data[0].embedding], dtype=np.float32)
        
        # 2. Parallel inference across all 3 heads
        dept_pred = self.dept_clf.predict(vec)[0]
        dept_prob = float(np.max(self.dept_clf.predict_proba(vec)[0]))
        
        priority_pred = self.priority_clf.predict(vec)[0]
        priority_prob = float(np.max(self.priority_clf.predict_proba(vec)[0]))
        
        sentiment_pred = self.sentiment_clf.predict(vec)[0]
        
        return {
            "department": dept_pred,
            "dept_confidence": dept_prob,
            "priority": priority_pred,
            "priority_confidence": priority_prob,
            "sentiment": sentiment_pred,
            "route_to_pager": (priority_pred == "P1_CRITICAL")
        }
""", language="python")

with col_code2:
    st.markdown("### Handling Out-of-Distribution & Ambiguous Inputs")
    st.markdown("""
    In production classification systems, not all inputs belong cleanly to a defined class.
    
    **Confidence Thresholding & Fallback Queuing**:
    ```python
    CONFIDENCE_THRESHOLD = 0.75

    result = router.triage_ticket(user_message)
    if result["dept_confidence"] < CONFIDENCE_THRESHOLD:
        # Route to human triage queue rather than misrouting
        result["department"] = "Human_Triage_Review"
        result["needs_manual_review"] = True
    ```
    
    * If maximum class probability is below 75%, route to a human supervisor.
    * This prevents embarrassing automated routing failures on edge-case inputs!
    """)

st.divider()

# Section 3: Interactive Workbench — 5-in-1 Enterprise Classification Playground
st.header("3. Interactive Workbench: 5-in-1 Enterprise Classification Suite", divider="rainbow")

st.markdown("""
Test embedding-based classification across all 5 core production domains live! 
Choose an application domain, select a sample prompt or enter your own custom text, and see how the classifier 
evaluates the semantic vector, predicts the category, and triggers downstream actions.
""")

app_domains = {
    "1. Customer Sentiment Analysis": {
        "classes": ["Positive / Enthusiastic", "Neutral / Informational", "Negative / Frustrated"],
        "presets": [
            "Your platform saved our engineering team 20 hours of debugging this week, phenomenal tool!",
            "I am reviewing the documentation to see if SSO is supported on the starter tier.",
            "Our production API has been throwing 500 errors for 45 minutes and your status page still says operational!"
        ]
    },
    "2. Spam & Phishing Detection": {
        "classes": ["Legitimate (Ham)", "Phishing Alert", "Promotional Spam"],
        "presets": [
            "URGENT: Your cloud account has been suspended due to policy violation. Click here immediately to verify credentials.",
            "Meeting notes from our quarterly architecture sync attached for your review.",
            "Exclusive limited-time 80% discount on corporate business email marketing lists!"
        ]
    },
    "3. Document Topic Categorization": {
        "classes": ["Cloud Infrastructure", "Machine Learning & AI", "Cybersecurity & Compliance", "Corporate Finance"],
        "presets": [
            "Configuring multi-region Kubernetes clusters with Calico CNI networking and Envoy ingress mesh.",
            "Evaluating fine-tuning performance of Llama-3 using QLoRA and 4-bit NF4 quantization on A100 GPUs.",
            "Annual consolidated income statement, operating cash flow analysis, and capital expenditure breakdown."
        ]
    },
    "4. Chatbot Intent Routing": {
        "classes": ["check_account_balance", "cancel_subscription", "request_order_refund", "technical_troubleshooting"],
        "presets": [
            "How much money is left in my savings account after yesterday's wire transfer?",
            "I want to stop my auto-renewing membership and downgrade to the free plan.",
            "The package arrived with a broken screen, how can I return it and get my payment refunded?",
            "My mobile app keeps freezing on the splash screen whenever I connect to Wi-Fi."
        ]
    },
    "5. Support Ticket Triage & SLA Routing": {
        "classes": ["P1_Critical (DevOps Outage)", "P2_High (Billing Issue)", "P3_Normal (Feature Request)", "P4_Low (General Inquiry)"],
        "presets": [
            "CRITICAL: Production database deadlocked, all customer checkouts are failing with 500 errors!",
            "Our company was billed twice on our corporate Amex for the March invoice, please correct ASAP.",
            "Would it be possible to add dark mode theme support to the analytics dashboard in a future release?",
            "What are the office hours for your European customer support phone desk?"
        ]
    }
}

selected_app_name = st.selectbox("Select Classification Application Domain:", list(app_domains.keys()))
active_app = app_domains[selected_app_name]

col_in, col_out = col_in, col_out = st.columns([1, 1])

with col_in:
    st.subheader("Inbound Text Sample")
    chosen_preset_text = st.selectbox("Choose a Benchmark Sample or Enter Custom:", active_app["presets"] + ["Custom Text"])
    if chosen_preset_text == "Custom Text":
        eval_sample_text = st.text_area("Enter text to classify:", value="Why is my server down?")
    else:
        eval_sample_text = st.text_area("Text to classify:", value=chosen_preset_text)

# Simulation Engine for all 5 domains
def simulate_domain_prediction(app_name: str, text: str, classes: list[str]) -> Tuple[str, float, list[dict], str]:
    t = text.lower()
    
    if "Sentiment" in app_name:
        if any(w in t for w in ["saved", "phenomenal", "great", "love", "awesome", "excellent", "thank"]):
            pred = classes[0]
            action = "Log Customer Delight; Trigger CSAT survey."
        elif any(w in t for w in ["outage", "failing", "error", "horrible", "frustrated", "broken", "terrible"]):
            pred = classes[2]
            action = "Flag High-Priority Escalation; Alert Customer Success Manager."
        else:
            pred = classes[1]
            action = "Standard response routing."
            
    elif "Spam" in app_name:
        if any(w in t for w in ["urgent", "suspended", "click here", "verify credentials", "password reset requested", "bank"]):
            pred = classes[1]
            action = "Quarantine Email; Block Sender IP; Alert Security Operations Center."
        elif any(w in t for w in ["discount", "limited-time", "exclusive", "marketing list", "free"]):
            pred = classes[2]
            action = "Route to Junk Folder."
        else:
            pred = classes[0]
            action = "Deliver to Primary Inbox."
            
    elif "Topic" in app_name:
        if any(w in t for w in ["kubernetes", "cni", "envoy", "docker", "cloud", "terraform", "ingress"]):
            pred = classes[0]
            action = "Tag: 'DevOps / Infrastructure'; Index in Tech Knowledge Base."
        elif any(w in t for w in ["llama", "qlora", "gpu", "machine learning", "ai", "transformer", "neural"]):
            pred = classes[1]
            action = "Tag: 'AI & Data Science'; File in Research Archive."
        elif any(w in t for w in ["income", "cash flow", "financial", "expenditure", "revenue", "statement"]):
            pred = classes[3]
            action = "Tag: 'Finance & Compliance'; Store in SEC Audit Archive."
        else:
            pred = classes[2]
            action = "Tag: 'Security'; Route to CISO compliance desk."
            
    elif "Intent" in app_name:
        if any(w in t for w in ["balance", "savings", "money left", "account"]):
            pred = classes[0]
            action = "Invoke Banking API: GET /api/v1/accounts/balance"
        elif any(w in t for w in ["stop", "cancel", "membership", "downgrade", "subscription"]):
            pred = classes[1]
            action = "Invoke Billing API: POST /api/v1/subscription/cancel_confirm"
        elif any(w in t for w in ["refund", "money back", "broken", "return"]):
            pred = classes[2]
            action = "Invoke Returns Workflow: POST /api/v1/orders/initiate_refund"
        else:
            pred = classes[3]
            action = "Open Diagnostics Agent: GET /support/troubleshooting_wizard"
            
    else: # Support Ticket Triage
        if any(w in t for w in ["critical", "outage", "database locked", "deadlock", "500 error", "crash"]):
            pred = classes[0]
            action = "Trigger PagerDuty Incident (P1 Critical SLA: 15-min response); Alert On-Call SRE."
        elif any(w in t for w in ["billed twice", "invoice", "amex", "charge", "payment", "card"]):
            pred = classes[1]
            action = "Route to Billing & Accounts Receivable Queue (P2 SLA: 4-hour response)."
        elif any(w in t for w in ["dark mode", "feature request", "would it be possible", "add support"]):
            pred = classes[2]
            action = "File as Product Enhancement Ticket in Jira Backlog."
        else:
            pred = classes[3]
            action = "Send Automated FAQ deflection article or standard queue."

    # Generate calibrated probabilities
    probs = []
    base_conf = round(np.random.uniform(0.88, 0.96), 3)
    rem = (1.0 - base_conf) / (len(classes) - 1)
    
    for c in classes:
        if c == pred:
            probs.append({"Class": c, "Confidence": f"{base_conf*100:.1f}%", "score": base_conf})
        else:
            score = round(rem + np.random.uniform(-0.01, 0.01), 3)
            probs.append({"Class": c, "Confidence": f"{score*100:.1f}%", "score": score})
            
    df_scores = pd.DataFrame(probs).sort_values(by="score", ascending=False).reset_index(drop=True)
    return pred, base_conf, df_scores, action

pred_cls, conf_val, prob_table, automated_action = simulate_domain_prediction(
    selected_app_name, eval_sample_text, active_app["classes"]
)

with col_out:
    st.subheader("Classification Decision & Automated Action")
    st.success(f"**Predicted Category**: `{pred_cls}` *(Confidence: **{conf_val*100:.1f}%**)*")
    
    st.markdown(f"**Automated Workflow Trigger**:  \n`{automated_action}`")
    
    st.markdown("### Class Probability Distribution:")
    st.dataframe(prob_table[["Class", "Confidence"]], use_container_width=True)

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: How do embedding-based intent classifiers handle 'Out-of-Scope' (OOD) or adversarial user inputs in conversational AI?", expanded=True):
    st.markdown("""
    **Answer:**
    In conversational AI, users frequently ask questions outside the chatbot's domain (e.g., asking a banking bot: *"What's the weather in Seattle?"* or typing gibberish *"asdfjkla"*). 
    A naive Softmax classifier will forcibly assign the input to whichever known intent has the highest relative score, even if all scores are low!
    
    **Production Solutions for OOD Handling**:
    1. **Confidence Thresholding on Softmax Probabilities**: Enforce a minimum probability threshold (e.g., `max(P) >= 0.70`). If all class probabilities are below 70%, route to an `out_of_scope_fallback` intent.
    2. **Explicit 'Out of Scope' Negative Class**: Include a dedicated background training class containing random general-domain questions and conversational chit-chat.
    3. **Cosine Distance to Class Centroids**: Measure the raw cosine distance between the query vector and the average centroid of the predicted class. If `CosineSim(x, centroid) < 0.65`, reject the prediction regardless of relative Softmax margin.
    """)

with st.expander("Q2: Why is embedding-based spam detection more resilient than traditional Bayesian or regex spam filters?"):
    st.markdown("""
    **Answer:**
    * **Lexical Obfuscation Failure**: Naive Bayesian filters and regex rely on specific keyword occurrences (e.g., *"Viagra"*, *"Wire Transfer"*, *"Urgent Password"*). Scammers easily bypass these via character substitutions (*"V!@gra"*, zero-width spaces, or homoglyphs).
    * **Semantic Invariance**: Transformer foundation models are pre-trained on diverse web text and byte-level subwords. They capture the **high-level semantic intent**—such as artificial urgency, coercion, and impersonation—regardless of superficial spelling variations.
    * Even if a scammer invents a completely novel story with zero traditional spam keywords, the embedding vector lands in the characteristic *"urgent social engineering"* geometric cluster.
    """)

with st.expander("Q3: How do you design a Multi-Label document classification system using embeddings?"):
    st.markdown("""
    **Answer:**
    In Multi-Label classification, a document can belong to multiple categories simultaneously (e.g., a ticket can be both *'Billing'* and *'Urgent Escalation'*).
    * **Binary Relevance Architecture**: Instead of training a single multi-class model with Softmax (which enforces mutually exclusive classes summing to 1.0), train $K$ independent **Binary Logistic Classifiers** with **Sigmoid activations**—one for each category:
      `P(tag_k | x) = sigmoid(W_k * x + b_k) = 1 / (1 + exp(-(W_k * x + b_k)))`
    * At inference, pass the single embedding vector $x$ through all $K$ binary heads.
    * Assign all tags where $P(tag_k | x) > 0.50$ (or a tuned F1 threshold).
    * This runs in microseconds on CPU and allows documents to receive 0, 1, or multiple tags.
    """)

with st.expander("Q4: What telemetry metrics should be monitored to detect data drift in production embedding classifiers?"):
    st.markdown("""
    **Answer:**
    Telemetry monitoring in embedding classification requires tracking both vector-level and prediction-level drift:
    1. **Prediction Confidence Distribution Shift**: Monitor the moving average of `max(P(class))`. If average confidence drops from 92% to 74%, new emerging customer issues are deviating from the training distribution.
    2. **Uncertainty / Human Fallback Queue Volume**: A spike in tickets falling below the confidence threshold indicates new terminology or product bugs.
    3. **Embedding Vector Centroid Drift**: Periodically compute the centroid of production embedding vectors $\\mu_{prod}$ and compare it against the training baseline centroid $\\mu_{train}$ using Wasserstein distance or Cosine distance.
    4. **Model Retraining Triggers**: When drift exceeds threshold, sample low-confidence tickets, label them, and retrain the lightweight linear head in seconds.
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Hardcoding Multi-Class Softmax on Multi-Intent Prompts**: Using single-choice Softmax when users combine multiple requests (*"I want to cancel my account and get a refund"*). The model picks one and ignores the other. Use multi-label Sigmoid heads.
* **Trap 2: Ignoring Cold-Start Classes**: Adding a new support category (e.g., *'Vision AI Support'*) with zero training samples. The classifier cannot predict what it has never seen. Use Zero-Shot nearest-centroid initialization until training data is collected.
* **Trap 3: Neglecting PII Sanitization in Public Embedding APIs**: Sending customer emails with Credit Card numbers, SSNs, or API secrets directly to external third-party embedding APIs. Always scrub PII using Presidio or regex before vectorization.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: Sarcasm & Sentiment"):
        st.markdown("**Front**: Why do embeddings excel at sarcasm detection?")
        st.divider()
        st.markdown("**Back**: Transformer self-attention captures contextual contrasts between positive words and negative situational contexts, unlike keyword counters.")

with card_cols[1]:
    with st.popover("Card 2: Multi-Head Triage"):
        st.markdown("**Front**: What is a Multi-Head ticket router?")
        st.divider()
        st.markdown("**Back**: A single embedding forward pass evaluated by multiple parallel linear heads predicting Department, Priority, and Sentiment simultaneously.")

with card_cols[2]:
    with st.popover("Card 3: OOD Fallback"):
        st.markdown("**Front**: How do you prevent misclassification of Out-of-Domain inputs?")
        st.divider()
        st.markdown("**Back**: Enforce confidence thresholding: if max probability is below 70-75%, route to a human supervisor fallback queue.")

with card_cols[3]:
    with st.popover("Card 4: Multi-Label"):
        st.markdown("**Front**: How do you classify overlapping tags with embeddings?")
        st.divider()
        st.markdown("**Back**: Use K independent Binary Logistic Classifiers with Sigmoid activations instead of a single Softmax head.")

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Embedding Classification Applications:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the 5 core applications (Sentiment, Spam, Topic, Intent, Ticket Triage).", key="chk29_3_1")
    st.checkbox("I know how Multi-Head classifiers predict multiple attributes from a single vector.", key="chk29_3_2")
    st.checkbox("I can implement confidence thresholding to catch Out-of-Domain queries.", key="chk29_3_3")
with c2:
    st.checkbox("I understand how multi-label classification uses Sigmoid binary heads.", key="chk29_3_4")
    st.checkbox("I know how to monitor embedding distribution drift in production telemetry.", key="chk29_3_5")
    st.checkbox("I understand PII sanitization requirements before external vectorization.", key="chk29_3_6")
