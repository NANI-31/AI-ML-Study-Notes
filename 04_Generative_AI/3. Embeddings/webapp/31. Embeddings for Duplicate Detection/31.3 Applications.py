import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="31.3 Applications of Duplicate Detection - Embeddings Curriculum",
    page_icon="🛡️",
    layout="wide"
)

# Title & Overview
st.title("31.3 Applications: Duplicate Questions, Support Tickets, Documents & E-Commerce Products")
st.caption("Chapter 31: Embeddings for Duplicate Detection | Domain: 3. Vector Embeddings")
st.markdown("""
Duplicate detection is not merely an academic exercise—it is a critical data hygiene and cost optimization engine powering 
the world's largest consumer and enterprise platforms:

* **Duplicate Question Detection (Quora & Stack Overflow)**: Merging redundant community inquiries into canonical threads, concentrating answers and preventing fragmented knowledge.
* **Support-Ticket Incident Merging (Zendesk & ServiceNow)**: Automatically detecting that 500 incoming customer tickets are all reporting the same underlying outage, linking them to a single **Master Incident** to prevent agent duplicate work.
* **Document Deduplication for LLM Pre-Training (Common Crawl & FineWeb)**: Removing duplicated and mirrored web pages from multi-terabyte training corpora, which prevents LLMs from memorizing redundant text and saves millions of dollars in GPU training compute.
* **E-Commerce Catalog Deduplication (Amazon & Shopify)**: Identifying identical physical products sold by different merchants under slightly different titles, enabling unified product buy-boxes and price comparisons.
""")

st.divider()

# Section 1: The Four Production Application Blueprints
st.header("1. Production Blueprints for the Four Core Applications", divider="rainbow")

col_app1, col_app2 = col_app1, col_app2 = st.columns([1, 1])

with col_app1:
    st.markdown("""
    ### 1. Community Q&A Question Deduplication
    * **The Problem**: On platforms like Stack Overflow or Quora, thousands of users ask the same question using different words:
      * Question A: *"How do I iterate over rows in a Pandas DataFrame?"*
      * Question B: *"Best way to loop through Pandas rows and columns?"*
    * **Workflow**: As the user types their title, the platform computes its embedding, queries the question vector database, and intercepts submission with: *"Does this already answer your question?"*
    * **Impact**: Deflects up to **35% of redundant question submissions**.
    
    ### 2. Helpdesk Ticket Merging & Deflection
    * **The Problem**: During a payment gateway outage, 1,000 customers submit tickets within 15 minutes.
    * **Workflow**: Inbound ticket embeddings are matched against active incident clusters. If similarity $> 0.88$, the ticket is linked to the **Master Outage Incident**.
    * **Impact**: When the outage is resolved, closing the Master Ticket automatically updates and notifies all 1,000 customers in a single click!
    """)

with col_app2:
    st.markdown("""
    ### 3. LLM Pre-Training Web Corpus Deduplication
    * **The Problem**: Crawled web datasets (e.g., Common Crawl) contain up to **50% duplicated text** (scraped mirrors, terms of service headers, boilerplate disclaimers).
    * **Impact on LLMs**: Training models on duplicated text causes catastrophic memorization, degrades reasoning, and inflates training compute.
    * **Workflow**: Two-tier deduplication: MinHash + LSH filters near-duplicates; dense embeddings purge subtle semantic mirrors.
    
    ### 4. E-Commerce Multi-Merchant Product Matcher
    * **The Problem**: On multi-seller marketplaces, 20 different merchants list the *Sony WH-1000XM5 Headphones* with varying titles, abbreviations, and descriptions:
      * Merchant 1: *"Sony WH1000XM5/B Premium Noise Cancelling Wireless Headphones - Black"*
      * Merchant 2: *"Sony Over-Ear Bluetooth Headset XM5 Noise Cancellation"*
    * **Workflow**: Multimodal text + image embeddings identify identical UPC/ASIN product entities to power unified product pages.
    """)

st.markdown("### Production Ticket Auto-Merge Flow")
merge_dot = """
digraph TicketMerge {
    rankdir=LR;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    NewTicket [label="Inbound Ticket #9482\\n'Credit card payment failing on checkout'", fillcolor="#FEE2E2", color="#EF4444", style="filled,bold"];
    
    VectorANN [label="Vector DB HNSW Search\\nAgainst Active Incident Embeddings\\n(Latency: 3ms)", fillcolor="#DBEAFE"];

    MasterIncident [label="Master Incident #8801 (P1 Outage)\\n'Stripe Payment Gateway Downtime'\\n(Already linked: 342 tickets)", fillcolor="#FEF3C7", color="#F59E0B", style="filled,bold"];

    AutoMergeAction [label="Auto-Merge & Link\\nAssigns Ticket to Master Queue\\nSets Auto-Responder", fillcolor="#D1FAE5", color="#10B981", style="filled,bold"];

    NewTicket -> VectorANN;
    VectorANN -> MasterIncident [label="Cosine Sim = 0.932"];
    MasterIncident -> AutoMergeAction;
}
"""
st.graphviz_chart(merge_dot, use_container_width=False)

st.divider()

# Section 2: Code Implementation: Real-Time Ticket Deduplicator & Merger
st.header("2. Code Implementation: The Real-Time Incident Linker", divider="rainbow")

st.code("""
import numpy as np
from openai import OpenAI
import chromadb

client = OpenAI()
chroma_client = chromadb.Client()
active_incidents_col = chroma_client.get_or_create_collection(
    name="active_incidents",
    metadata={"hnsw:space": "cosine"}
)

SIMILARITY_THRESHOLD = 0.88

def get_embedding(text: str) -> list[float]:
    res = client.embeddings.create(model="text-embedding-3-small", input=text)
    return res.data[0].embedding

def triage_inbound_support_ticket(ticket_id: str, ticket_text: str) -> dict:
    \"\"\"Evaluates inbound ticket against active master incidents.\"\"\"
    ticket_vec = get_embedding(ticket_text)
    
    # Query vector database for active master outages
    results = active_incidents_col.query(
        query_embeddings=[ticket_vec],
        n_results=1,
        include=["metadatas", "distances"]
    )
    
    if results['ids'] and len(results['ids'][0]) > 0:
        # Cosine distance in Chroma: dist = 1 - cosine_sim
        cosine_dist = results['distances'][0][0]
        cosine_sim = 1.0 - cosine_dist
        master_incident_id = results['ids'][0][0]
        master_metadata = results['metadatas'][0][0]
        
        if cosine_sim >= SIMILARITY_THRESHOLD:
            # Automatically link to master incident
            return {
                "action": "AUTO_MERGE_TO_MASTER",
                "master_incident_id": master_incident_id,
                "incident_title": master_metadata["title"],
                "similarity": round(cosine_sim, 3),
                "resolution": "Linked to active outage thread. Suppressed from manual agent dispatch."
            }
            
    # No active master duplicate found -> route as independent ticket
    return {
        "action": "ROUTE_TO_STANDARD_DISPATCH",
        "ticket_id": ticket_id,
        "similarity": 0.0,
        "resolution": "New unique inquiry. Assigned to general tier-1 support queue."
    }
""", language="python")

st.divider()

# Section 3: Interactive Workbench — 4-in-1 Enterprise Deduplication Suite
st.header("3. Interactive Workbench: 4-in-1 Enterprise Deduplication Suite", divider="rainbow")

st.markdown("""
Test real-time duplicate detection across all 4 production applications live. 
Choose a domain, select an inbound query sample or type your own text, and observe how the system evaluates 
against the active repository, detects semantic clones, and triggers automated business resolutions!
""")

app_repositories = {
    "1. Community Q&A Question Deflection (Stack Overflow)": {
        "description": "Intercepts redundant developer questions and redirects them to verified canonical answers.",
        "active_database": [
            {"id": "QA_101", "title": "How to iterate over rows in a Pandas DataFrame?", "answers": 42, "accepted": True},
            {"id": "QA_102", "title": "How to convert a string to integer in Python with error handling?", "answers": 18, "accepted": True},
            {"id": "QA_103", "title": "Docker container exits immediately with code 0 after startup", "answers": 25, "accepted": True}
        ],
        "test_inbound": [
            "Best practice way to loop through rows and columns in a pandas dataframe?",
            "How do I parse a string into an int in python without throwing exceptions?",
            "Why does my docker container shutdown right after running docker run?"
        ]
    },
    "2. Customer Support Incident Merging (Zendesk)": {
        "description": "Detects high-volume customer complaints and auto-links them to active Sev-1 outage tickets.",
        "active_database": [
            {"id": "INC_901", "title": "Master Incident: Stripe Payment Gateway 504 Timeouts on Checkout", "answers": 150, "accepted": True},
            {"id": "INC_902", "title": "Master Incident: European Availability Zone Network Latency Spike", "answers": 84, "accepted": True}
        ],
        "test_inbound": [
            "Customer checkout is broken, credit card processing button hangs forever!",
            "Our servers in Frankfurt and Dublin are experiencing massive latency spikes.",
            "Can you tell me how to update our billing address on our PDF receipts?"
        ]
    },
    "3. E-Commerce Multi-Merchant Product Matcher": {
        "description": "Matches multi-seller catalog listings to unified product buy-boxes.",
        "active_database": [
            {"id": "ASIN_B01", "title": "Sony WH-1000XM5 Wireless Industry Leading Noise Cancelling Headphones - Black", "answers": 99, "accepted": True},
            {"id": "ASIN_B02", "title": "Apple MacBook Pro 16-inch M3 Max (36GB Unified Memory, 1TB SSD) - Space Black", "answers": 55, "accepted": True}
        ],
        "test_inbound": [
            "Sony WH1000XM5 Bluetooth Over-Ear ANC Headset - Midnight Black Edition",
            "MacBook Pro 16 M3 Max 36GB RAM 1TB Storage Space Black Laptop",
            "Standard wooden ergonomic footrest for under-desk office use"
        ]
    }
}

selected_app_domain = st.selectbox("Select Production Application Domain:", list(app_repositories.keys()))
active_repo = app_repositories[selected_app_domain]

st.info(f"**Domain Mission**: {active_repo['description']}")

col_test_in, col_test_out = col_test_in, col_test_out = st.columns([1, 1])

with col_test_in:
    st.subheader("Inbound Query / Submission")
    preset_chosen = st.selectbox("Choose a Test Inbound Sample or Enter Custom:", active_repo["test_inbound"] + ["Custom Input"])
    if preset_chosen == "Custom Input":
        inbound_eval_text = st.text_input("Enter custom text:", value="How to iterate over rows in pandas?")
    else:
        inbound_eval_text = st.text_input("Active Submission:", value=preset_chosen)
        
    st.markdown("### Existing Canonical Database Entries:")
    for item in active_repo["active_database"]:
        st.caption(f"• `[{item['id']}]` **{item['title']}**")

# Simulated Matcher
def evaluate_duplicate_application(domain_name: str, query_text: str, database: list[dict]):
    q_low = query_text.lower()
    
    best_match = None
    best_score = 0.0
    
    for item in database:
        t_low = item["title"].lower()
        score = 0.0
        
        # Keyword & concept overlap heuristics
        if "pandas" in q_low and "pandas" in t_low and ("iterate" in q_low or "loop" in q_low):
            score = 0.935
        elif "string" in q_low and "int" in q_low and "python" in q_low:
            score = 0.920
        elif "docker" in q_low and ("exit" in q_low or "shutdown" in q_low or "exits" in t_low):
            score = 0.910
        elif "checkout" in q_low or "payment" in q_low or "credit card" in q_low:
            if "stripe" in t_low or "payment" in t_low:
                score = 0.940
        elif "latency" in q_low or "frankfurt" in q_low or "european" in q_low:
            if "european" in t_low or "latency" in t_low:
                score = 0.925
        elif "wh1000xm5" in q_low or "sony" in q_low:
            if "wh-1000xm5" in t_low:
                score = 0.955
        elif "macbook" in q_low and "m3" in q_low:
            if "macbook" in t_low:
                score = 0.960
        else:
            score = round(np.random.uniform(0.20, 0.45), 3)
            
        if score > best_score:
            best_score = score
            best_match = item
            
    is_duplicate = (best_score >= 0.88)
    return is_duplicate, best_score, best_match

is_dup, match_score, matched_entity = evaluate_duplicate_application(
    selected_app_domain, inbound_eval_text, active_repo["active_database"]
)

with col_test_out:
    st.subheader("Deduplication Decision & Resolution")
    
    if is_dup:
        st.success(f"**DUPLICATE IDENTIFIED!** *(Match Score: **{match_score:.3f}**)* :white_check_mark:")
        st.markdown(f"**Matched Canonical Master**: `[{matched_entity['id']}]`  \n**Title**: *{matched_entity['title']}*")
        
        if "Community Q&A" in selected_app_domain:
            st.info("""
            **Automated Action**: **DEFLECTION INTERCEPT TRIGGERED!**  
            User prompt displayed: *"We found an existing community question with verified answers that solves your inquiry! Redirecting to [Answer #QA_101]..."*
            """)
        elif "Customer Support" in selected_app_domain:
            st.error("""
            **Automated Action**: **AUTO-MERGE TO ACTIVE SEV-1 OUTAGE!**  
            Ticket linked to `INC_901`. Outage status page notification sent to user. Ticket suppressed from agent queue.
            """)
        else:
            st.success("""
            **Automated Action**: **MULTI-SELLER PRODUCT MATCH!**  
            Listing merged into canonical catalog entity `ASIN_B01`. Merchant added as secondary seller in unified buy-box.
            """)
    else:
        st.info(f"**UNIQUE ITEM (Not a Duplicate)** *(Highest Cosine Match: {match_score:.3f})*")
        st.markdown("Action: No duplicate found above 0.88 threshold. Permitted as a fresh, standalone submission.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: How do platforms like Quora and Stack Overflow deflect duplicate question submissions using embeddings in real-time?", expanded=True):
    st.markdown("""
    **Answer:**
    Platforms deploy a **Real-Time Client-Side Deflection Pipeline**:
    1. **As-You-Type Vectorization**: As the user types their question title in the input box, an event listener debounces keystrokes (e.g., 300ms pause).
    2. **Low-Latency ANN Query**: The frontend sends the title string to an inference microservice. A lightweight, distilled embedding model (e.g., MiniLM or BGE-small) generates a vector in sub-10ms.
    3. **Vector Database Lookup**: The vector is searched against the pre-indexed HNSW index of answered questions with threshold `CosineSim >= 0.88`.
    4. **Deflection Intercept UI**: If matches are found, the UI renders dynamic preview cards: *"This question looks identical to: [How to iterate rows in pandas? (42 answers)]. Does this solve your problem?"*
    5. If the user clicks "Yes", the submission is aborted, deflecting redundant database records and instantly satisfying the user's intent.
    """)

with st.expander("Q2: In e-commerce catalog matching, why is text embedding alone often insufficient, and how are multimodal embeddings applied?"):
    st.markdown("""
    **Answer:**
    In multi-seller e-commerce (Amazon, eBay, Walmart):
    * **Text Overlap Ambiguity**: A seller listing an *"iPhone 15 Case"* might use a title that includes *"For Apple iPhone 15 Black Leather"*, which has high text cosine similarity to the actual *"Apple iPhone 15"* smartphone listing!
    * Relying solely on text embeddings leads to catastrophic false-positive merges between accessories and actual hardware.
    * **The Multimodal Solution**:
      1. **Text Embedding (Title + Specs)**: Matches brand, model number, and technical specifications.
      2. **Vision Embedding (CLIP / DINOv2 on product photos)**: Compares the main product thumbnail. A phone case image vector has a very low visual similarity to an actual smartphone screen!
      3. **Hard Attribute Filtering**: Strict deterministic filtering on UPC, EAN, or manufacturer part numbers (MPN) where available.
      4. Merging only occurs when both text similarity $> 0.88$ AND visual similarity $> 0.85$.
    """)

with st.expander("Q3: How do automated support ticket deduplicators handle customer follow-ups and repeat messages?"):
    st.markdown("""
    **Answer:**
    Customers often send multiple emails across hours (*"Any update on my issue?"*, *"Still waiting for a response"*).
    * **User-Scoped Thread Matching**: Before querying global incident databases, the pipeline first searches the specific user's open ticket history:
      `filter={"user_id": customer_id, "status": "open"}`
    * If an incoming email from the same user has high similarity or anaphoric follow-up intent to their existing open ticket, it is appended as an internal comment to the open thread rather than creating a duplicate ticket.
    * This prevents a single customer from spawning 5 distinct ticket queues for the same inquiry.
    """)

with st.expander("Q4: Why is deduplication crucial during LLM pre-training data preparation (such as FineWeb or Llama-3)?"):
    st.markdown("""
    **Answer:**
    Deduplicating web crawl text prior to LLM training is critical for three reasons:
    1. **Preventing Memorization & Privacy Leaks**: Models trained on duplicated text memorize verbatim paragraphs (including PII, credit card numbers, and copyright text) 10x more frequently than on deduplicated corpora.
    2. **Training Compute Efficiency**: Web crawls typically contain 30% to 50% duplicate or near-duplicate documents. Deduplicating removes billions of tokens, allowing the training budget to expose the model to diverse, unique knowledge.
    3. **Generalization & Downstream Accuracy**: Empirical research (Lee et al., 2022, *Deduplicating Training Data Makes Language Models Better*) demonstrated that models trained on deduplicated datasets achieve higher zero-shot accuracy and lower perplexity on downstream benchmarks.
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Intercepting Questions with Unanswered Duplicates**: Deflecting a user's question by redirecting them to an existing question that has **zero answers** or accepted resolutions. The user gets stuck in an unresolved dead end. Only deflect to questions with accepted or highly upvoted answers.
* **Trap 2: Auto-Closing Support Tickets Without Transparent Explanations**: Silently closing a customer's ticket with a generic 'Duplicate' status without informing them of the linked master outage ticket or status page link.
* **Trap 3: Merging Variants with Different SKUs**: In e-commerce, merging the *"128GB iPhone"* listing with the *"512GB iPhone"* listing as duplicates because the titles share 95% of the same words. Always extract structured storage and color attributes as deterministic blocking keys.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: Question Deflection"):
        st.markdown("**Front**: How does question deflection work in community forums?")
        st.divider()
        st.markdown("**Back**: Vectorizes the user's title in real-time as they type and intercepts submission with existing answered canonical questions.")

with card_cols[1]:
    with st.popover("Card 2: Master Incident Link"):
        st.markdown("**Front**: Why merge support tickets into a Master Incident?")
        st.divider()
        st.markdown("**Back**: Prevents hundreds of support agents from duplicating effort during outages; resolving the Master Ticket resolves all linked tickets simultaneously.")

with card_cols[2]:
    with st.popover("Card 3: LLM Pre-training Dedup"):
        st.markdown("**Front**: Why is deduplication essential for LLM training datasets?")
        st.divider()
        st.markdown("**Back**: Prevents verbatim memorization, eliminates 30-50% redundant training tokens, and improves reasoning generalization.")

with card_cols[3]:
    with st.popover("Card 4: Multimodal Catalog Match"):
        st.markdown("**Front**: Why combine text and image embeddings in e-commerce deduplication?")
        st.divider()
        st.markdown("**Back**: Prevents accessories (e.g., phone cases) from accidentally matching actual hardware devices based solely on shared text keywords.")

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Duplicate Detection Applications:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the 4 core applications of duplicate detection.", key="chk31_3_1")
    st.checkbox("I know how real-time question deflection intercepts redundant submissions.", key="chk31_3_2")
    st.checkbox("I can implement an automated support ticket linker to master outages.", key="chk31_3_3")
with c2:
    st.checkbox("I understand why LLM pre-training data requires strict deduplication.", key="chk31_3_4")
    st.checkbox("I know why multimodal embeddings prevent accessory-to-device false matches in e-commerce.", key="chk31_3_5")
    st.checkbox("I can design deterministic blocking filters (UPC, SKU, storage capacity).", key="chk31_3_6")
