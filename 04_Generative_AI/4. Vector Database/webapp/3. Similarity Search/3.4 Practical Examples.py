import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="3.4 Practical Examples - Vector DB Curriculum",
    page_icon="💼",
    layout="wide"
)

# Title & Overview
st.title("3.4 Practical Examples: Questions, Documents, Products, Images & Customer Profiles")
st.caption("Chapter 3: Similarity Search | Domain: 4. Vector Database")
st.markdown("""
Similarity search is the operational engine behind many of the most valuable machine learning systems in industry today. 
Whether deflecting redundant customer support tickets, uncovering precedent in thousands of legal contracts, 
recommending visually similar retail merchandise, detecting copyright violations, or assembling lookalike marketing cohorts, 
the underlying pipeline follows the same geometric principles.

In this lesson, we examine real-world system designs across the 5 canonical similarity search domains:
1. **Similar Questions** (FAQ Deduplication & Intent Matching)
2. **Similar Documents** (Legal Discovery & Knowledge Mining)
3. **Similar Products** (E-Commerce Recommendations & Two-Tower Encoders)
4. **Similar Images** (Visual Search & Reverse Image Retrieval)
5. **Similar Customer Profiles** (Lookalike Audiences & Churn Risk Segmentation)
""")

st.divider()

# Section 1: The 5 Canonical Production Domains
st.header("1. Architectural Breakdown Across 5 Enterprise Domains", divider="rainbow")

col_dom1, col_dom2 = st.columns([1, 1])

with col_dom1:
    st.markdown("""
    ### 1. Textual & Semantic Domains
    * **1. Similar Questions (Support Ticket Deflection)**:
      * *Challenge*: Customers phrase the same problem in thousands of ways: `"App crashes on launch"` vs `"Won't open after update"`.
      * *Design*: Encode incoming queries against a curated FAQ database using symmetric sentence encoders (`bge-base-en` or `all-mpnet-base-v2`). Deflect up to 40% of inbound tickets before agent assignment.
    * **2. Similar Documents (Contract & Document Discovery)**:
      * *Challenge*: Documents span 20 to 100 pages; semantic dilution destroys whole-document vectors.
      * *Design*: Parent-child chunking (embed 256-token paragraphs, link to parent sections). Compute paragraph-to-paragraph similarity to surface relevant precedent clauses.
    """)

with col_dom2:
    st.markdown("""
    ### 2. Multimodal & Behavioral Domains
    * **3. Similar Products (E-Commerce Two-Tower Recommenders)**:
      * *Design*: Two-Tower DNNs or multimodal models (text metadata + product image). Dot product between user vector and product vector yields sub-10ms candidate generation for catalog of 50M items.
    * **4. Similar Images (Reverse Visual Search)**:
      * *Design*: Vision Transformers (ViT, CLIP, DINOv2) map product photos or user camera uploads to 512-d or 768-d unit vectors. Nearest neighbor retrieval matches color, silhouette, texture, and style.
    * **5. Similar Customer Profiles (Fintech & Marketing Lookalikes)**:
      * *Design*: Graph neural networks (Node2Vec) or autoencoders project user transaction behavior, browsing frequency, and demographic vectors into behavioral embeddings for fraud rings and lookalike targeting.
    """)

st.markdown("### Domain Architecture Comparison Matrix")
domain_matrix_df = pd.DataFrame({
    "Domain Use Case": ["Similar Questions", "Similar Documents", "Similar Products", "Similar Images", "Similar Customer Profiles"],
    "Input Modality": ["Short Text (10-30 tokens)", "Long Text (1k-50k tokens)", "Hybrid (Catalog text + photo)", "Image Pixels (RGB)", "Tabular + Graph Sequences"],
    "Leading Encoder": ["all-MiniLM / BGE-small", "OpenAI v3 / BGE-M3 (Chunked)", "Two-Tower DNN / CLIP", "CLIP ViT-B/32 / DINOv2", "Autoencoder / Node2Vec / TabNet"],
    "Distance Metric": ["Cosine Similarity", "Cosine Distance", "Dot Product (Normalized)", "Cosine / Euclidean L2", "Euclidean (L2) / Cosine"],
    "Typical Scale": ["50k - 500k FAQs", "1M - 100M Chunks", "10M - 500M SKUs", "10M - 1B Images", "5M - 200M Users"],
    "Latency SLA": ["< 15 ms", "< 30 ms", "< 10 ms", "< 25 ms", "Batch or < 50 ms"]
})
st.dataframe(domain_matrix_df, use_container_width=True, hide_index=True)

st.divider()

# Section 2: Universal End-to-End Enterprise Architecture Flowchart
st.header("2. Universal Production Architecture Flowchart", divider="rainbow")

st.markdown("""
Regardless of modality, all 5 similarity search domains converge onto a unified retrieval and hydration pipeline:
""")

uni_arch_dot = """
digraph UniversalSimilarityFlow {
    rankdir=LR;
    size="5,3";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
    edge [fontname="Helvetica", fontsize=7];

    RawInput [label="User / Event Trigger\\nQuestion / Doc / Image / User ID", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
    Encoder [label="Domain Encoder\\n(NLP / ViT / Two-Tower / GNN)", fillcolor="#DBEAFE"];
    UnitNorm [label="Unit L2 Normalizer\\n||q|| = 1.0", fillcolor="#BFDBFE"];

    subgraph cluster_vdb {
        label="Vector Database Engine (HNSW / IVF-PQ)"; style="dashed"; color="#10B981";
        FilterMask [label="Metadata Filter Mask\\n(tenant, category, active)", fillcolor="#D1FAE5"];
        SIMDSearch [label="Top-K SIMD ANN Search\\nDot Product / Cosine", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];
    }

    Reranker [label="Cross-Encoder Reranker\\nHigh Precision Scoring", fillcolor="#FEF3C7"];
    HydratedPayload [label="Hydrated Output\\nFinal Actionable Results", fillcolor="#FDE68A", color="#D97706", style="filled,bold"];

    RawInput -> Encoder -> UnitNorm -> SIMDSearch;
    FilterMask -> SIMDSearch;
    SIMDSearch -> Reranker -> HydratedPayload;
}
"""
st.graphviz_chart(uni_arch_dot, use_container_width=False)

st.divider()

# Section 3: Interactive 5-Domain Similarity Search Workbench
st.header("3. Interactive 5-Domain Similarity Search Workbench", divider="rainbow")

st.markdown("""
Select any of the 5 canonical enterprise domains to test real-world similarity search queries against 
curated realistic domain catalogs:
""")

selected_domain = st.selectbox(
    "Choose Enterprise Domain to Simulate",
    options=[
        "1. Similar Questions (Support Ticket Deflection)",
        "2. Similar Documents (Contract Clause Discovery)",
        "3. Similar Products (E-Commerce Recommendations)",
        "4. Similar Images (Visual Merchandise Search)",
        "5. Similar Customer Profiles (Fintech Lookalike Cohorts)"
    ]
)

col_d_in, col_d_out = st.columns([1, 1])

if "1. Similar Questions" in selected_domain:
    with col_d_in:
        st.subheader("Support Ticket Input")
        q_ticket = st.selectbox(
            "Customer Inbound Ticket",
            options=[
                "My mobile app crashes right when I tap the login button",
                "How do I reset my account password if I lost access to my email?",
                "I was double charged on my monthly subscription invoice",
                "Can I export my project data to CSV format?"
            ]
        )
        threshold_q = st.slider("Deflection Confidence Threshold", 0.60, 0.95, 0.78, 0.02)
        
    with col_d_out:
        st.subheader("Matched Knowledge Base Articles")
        faq_db = [
            ("FAQ-101", "Troubleshooting iOS and Android app crashes during authentication", 0.92 if "crashes" in q_ticket else 0.35, "Article: Clear app cache and reinstall update v2.4."),
            ("FAQ-102", "Account recovery options when two-factor authentication or email is unavailable", 0.89 if "password" in q_ticket or "email" in q_ticket else 0.40, "Article: Submit identity verification selfie."),
            ("FAQ-103", "Resolving duplicate invoice billing and refund processing timelines", 0.94 if "charged" in q_ticket or "invoice" in q_ticket else 0.28, "Article: Automated refunds process in 3-5 business days."),
            ("FAQ-104", "Exporting workspace reports to Excel, CSV, and PDF formats", 0.91 if "CSV" in q_ticket or "export" in q_ticket else 0.31, "Article: Go to Settings > Export Data.")
        ]
        faq_db.sort(key=lambda x: x[2], reverse=True)
        top_art = faq_db[0]
        
        st.metric("Top Matched Article", top_art[0], f"Similarity: {top_art[2]:.2f}")
        if top_art[2] >= threshold_q:
            st.success(f"**Automatic Deflection Activated!**\n\n*Matched:* `{top_art[1]}`\n\n*Solution:* {top_art[3]}")
        else:
            st.warning("Below confidence threshold. Routing to human tier-2 support agent.")

elif "2. Similar Documents" in selected_domain:
    with col_d_in:
        st.subheader("Contract Clause Query")
        clause_query = st.selectbox(
            "Target Legal Clause Requirement",
            options=[
                "Indemnification for third-party intellectual property infringement claims",
                "Limitation of liability capped at aggregate fees paid over 12 months",
                "Termination for convenience upon thirty days written advance notice"
            ]
        )
    with col_d_out:
        st.subheader("Discovered Historical Contract Precedents")
        contracts = [
            ("MSA-Vendor-2023", "Provider shall indemnify, defend, and hold harmless against third-party IP claims...", 0.93 if "intellectual" in clause_query else 0.45),
            ("SaaS-Agreement-2024", "Neither party's total cumulative liability shall exceed total fees paid in preceding twelve months...", 0.91 if "liability" in clause_query else 0.38),
            ("Vendor-SOW-2022", "Either party may terminate this agreement without cause upon thirty (30) days prior written notice...", 0.89 if "Termination" in clause_query else 0.41)
        ]
        contracts.sort(key=lambda x: x[2], reverse=True)
        for cid, ctext, csim in contracts:
            st.markdown(f"**{cid}** | Cosine Sim: `{csim:.2f}`")
            st.markdown(f"*Clause Snippet:* `{ctext}`")
            st.markdown("---")

elif "3. Similar Products" in selected_domain:
    with col_d_in:
        st.subheader("Seed Product")
        seed_prod = st.selectbox(
            "Customer Currently Viewing SKU",
            options=[
                "Sony WH-1000XM5 Wireless Noise Canceling Headphones ($399)",
                "Apple MacBook Pro 16-inch M3 Max 36GB RAM ($3,499)",
                "Ergonomic Mesh Office Chair with Adjustable Lumbar Support ($249)"
            ]
        )
        cat_filter = st.checkbox("Filter by Same Department Category", value=True)
    with col_d_out:
        st.subheader("Recommended Related Products (Top-K)")
        prods = [
            ("SKU-9901", "Bose QuietComfort Ultra Headphones ($429)", "Electronics", 0.91 if "Headphones" in seed_prod else 0.45),
            ("SKU-9902", "Sennheiser Momentum 4 Wireless ANC ($379)", "Electronics", 0.88 if "Headphones" in seed_prod else 0.40),
            ("SKU-9903", "Dell XPS 16 Laptop Intel Core Ultra 9 ($3,199)", "Computers", 0.89 if "MacBook" in seed_prod else 0.30),
            ("SKU-9904", "Steelcase Gesture Ergonomic Office Chair ($1,399)", "Furniture", 0.92 if "Chair" in seed_prod else 0.20)
        ]
        prods.sort(key=lambda x: x[3], reverse=True)
        for pid, pname, pcat, psim in prods[:3]:
            st.markdown(f"**{pid}**: {pname} | Category: `{pcat}` | Affinity: `{psim:.2f}`")

elif "4. Similar Images" in selected_domain:
    with col_d_in:
        st.subheader("Visual Query Image Simulation")
        img_query = st.selectbox(
            "Uploaded Camera Photo",
            options=[
                "photo_vintage_leather_jacket_brown.jpg (Vintage Apparel)",
                "photo_modern_scandinavian_coffee_table.jpg (Home Decor)",
                "photo_trail_running_shoes_waterproof.jpg (Footwear)"
            ]
        )
    with col_d_out:
        st.subheader("Visual Reverse Search Catalog Hits")
        imgs = [
            ("IMG-401", "Distressed Brown Leather Biker Jacket", "Fashion", 0.93 if "jacket" in img_query else 0.25),
            ("IMG-402", "Suede Aviator Bomber Jacket in Cognac", "Fashion", 0.86 if "jacket" in img_query else 0.31),
            ("IMG-403", "Minimalist Oak Round Coffee Table", "Home", 0.91 if "table" in img_query else 0.15),
            ("IMG-404", "Gore-Tex All-Terrain Trail Running Shoe", "Footwear", 0.94 if "running" in img_query else 0.20)
        ]
        imgs.sort(key=lambda x: x[3], reverse=True)
        for iid, iname, icat, isim in imgs[:2]:
            st.markdown(f"**{iid}**: {iname} | `{icat}` | Visual Sim: `{isim:.2f}`")

else: # Customer Profiles
    with col_d_in:
        st.subheader("Seed Customer Profile")
        user_query_prof = st.selectbox(
            "Target Seed Customer Persona",
            options=[
                "High-Frequency Fintech Day Trader (Avg volume $50k/day, Mobile user)",
                "Budget Family Grocery Shopper (Coupons, weekend volume, price sensitive)",
                "Enterprise SaaS Developer (API heavy, evening activity, US West)"
            ]
        )
    with col_d_out:
        st.subheader("Lookalike Audience Cluster Candidates")
        cohort = [
            ("USER-8192", "Mobile Crypto/Stock Trader (Vol: $42k/day, Peak: 9:30 AM)", 0.94 if "Trader" in user_query_prof else 0.30),
            ("USER-8193", "Options & Equities Active Trader (Vol: $68k/day)", 0.91 if "Trader" in user_query_prof else 0.35),
            ("USER-4412", "Suburban Family Shopper (Bi-weekly grocery runs, high coupon usage)", 0.93 if "Grocery" in user_query_prof else 0.25),
            ("USER-1099", "Backend Cloud Engineer (High API usage, nocturnal work patterns)", 0.92 if "SaaS" in user_query_prof else 0.20)
        ]
        cohort.sort(key=lambda x: x[2], reverse=True)
        for uid, udesc, usim in cohort[:2]:
            st.markdown(f"**{uid}**: {udesc} | Lookalike Similarity: `{usim:.2f}`")

st.divider()

# Section 4: Executable Python Implementation
st.header("4. Executable Code: Multi-Domain Production Similarity Engine", divider="rainbow")

st.markdown("""
The following complete Python program implements a **Multi-Domain Vector Search System** that ingests 
heterogeneous entities (Questions, Documents, Products, Images, Users) and executes typed, filtered similarity retrieval:
""")

code_multidomain = '''import numpy as np
from typing import List, Dict, Any

class MultiDomainEntity:
    def __init__(self, entity_id: str, domain: str, vector: np.ndarray, metadata: Dict[str, Any]):
        self.entity_id = entity_id
        self.domain = domain
        # Normalize to unit sphere
        norm = np.linalg.norm(vector)
        self.vector = vector / (norm if norm > 0 else 1.0)
        self.metadata = metadata

class EnterpriseSimilarityEngine:
    """Production similarity search engine supporting multi-domain collections."""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.entities: Dict[str, MultiDomainEntity] = {}
        self.domain_indices: Dict[str, List[str]] = {}

    def insert(self, entity: MultiDomainEntity):
        self.entities[entity.entity_id] = entity
        if entity.domain not in self.domain_indices:
            self.domain_indices[entity.domain] = []
        self.domain_indices[entity.domain].append(entity.entity_id)

    def search_domain(
        self,
        domain: str,
        query_vector: np.ndarray,
        top_k: int = 3,
        threshold: float = 0.70
    ) -> List[Dict[str, Any]]:
        """Restricts vector search strictly to target domain entities."""
        if domain not in self.domain_indices:
            return []

        # Normalize query
        q_norm = np.linalg.norm(query_vector)
        q_vec = query_vector / (q_norm if q_norm > 0 else 1.0)

        # Build sub-matrix for domain
        candidate_ids = self.domain_indices[domain]
        matrix = np.array([self.entities[cid].vector for cid in candidate_ids])

        # SIMD dot product
        scores = np.dot(matrix, q_vec)

        # Rank
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices:
            sc = float(scores[idx])
            if sc < threshold:
                break
            cid = candidate_ids[idx]
            results.append({
                "id": cid,
                "score": sc,
                "metadata": self.entities[cid].metadata
            })
            if len(results) == top_k:
                break

        return results

# Demonstration
dim = 64
engine = EnterpriseSimilarityEngine(dimension=dim)
np.random.seed(101)

# Ingest across domains
# 1. Questions
engine.insert(MultiDomainEntity("FAQ_01", "questions", np.random.randn(dim), {"q": "How to reset password?"}))
engine.insert(MultiDomainEntity("FAQ_02", "questions", np.random.randn(dim), {"q": "How to change email address?"}))

# 2. Products
engine.insert(MultiDomainEntity("SKU_99", "products", np.random.randn(dim), {"name": "Wireless Noise Canceling Headphones"}))

# Query the 'questions' domain
q_vec = engine.entities["FAQ_01"].vector.copy() + np.random.randn(dim) * 0.10
hits = engine.search_domain("questions", q_vec, top_k=1, threshold=0.60)

print(f"Discovered {len(hits)} matching FAQ item(s):")
for h in hits:
    print(f"ID={h['id']} | Score={h['score']:.4f} | Payload={h['metadata']}")
'''

st.code(code_multidomain, language="python")

if st.button("🚀 Execute Multi-Domain Similarity Engine", type="primary"):
    exec_scope = {}
    exec(code_multidomain, globals(), exec_scope)
    hits_out = exec_scope["hits"]
    
    st.success("Multi-Domain Engine executed successfully!")
    for h in hits_out:
        st.markdown(f"`ID: {h['id']}` | **Cosine Similarity:** `{h['score']:.4f}`")
        st.json(h["metadata"])

st.divider()

# Section 5: Technical Interview Preparation & Production Traps
st.header("5. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. In question deduplication systems, why is Question-to-Question matching fundamentally different from Question-to-Answer matching?"""
a1 = """* **Question-to-Question (Q2Q)**:
  * Both inputs share identical semantic intent, syntax, and interrogative phrasing.
  * Best handled by symmetric sentence encoders (e.g. `all-MiniLM-L6-v2`, `bge-base-en`).
  * Yields higher precision because the goal is identifying whether two users are asking the exact same problem.
* **Question-to-Answer (Q2A)**:
  * Asymmetric retrieval: a short question (3-10 tokens) retrieves a long informative passage (200-500 tokens).
  * Requires asymmetric bi-encoders with instruction prefixes (`query:` vs `passage:`) or dual-tower architectures.
* **Production Best Practice**: Always store both the canonical question and the answer text in the vector database. Search user queries against canonical questions first (Q2Q) for deflection; fall back to Q2A for broader retrieval."""

q2 = """2. How do Two-Tower Neural Networks scale e-commerce product recommendations to hundreds of millions of items?"""
a2 = """* **Architecture**:
  * **User Tower**: Maps user historical click sequences, search queries, and demographic traits into a vector `u in R^d`.
  * **Item Tower**: Maps product images, catalog descriptions, brand, and pricing into a vector `v in R^d`.
* **Decoupled Offline Ingestion**:
  * Product embeddings `v` are computed asynchronously offline and indexed in an HNSW vector database.
* **Online Retrieval (< 10ms)**:
  * At query time, only the lightweight User Tower runs online to produce `u`.
  * A single vector database query `q = u` retrieves Top-100 product candidates in single-digit milliseconds, which are then passed to a secondary heavy cross-network ranking model."""

q3 = """3. What is the Cold-Start Problem in embedding-based product recommendations, and how do vector databases solve it?"""
a3 = """* In traditional collaborative filtering (matrix factorization), a newly listed product has zero user click/purchase history and cannot be recommended.
* **Embedding-Based Solution**:
  * Content-based multimodal embedding models (e.g., CLIP or text-image fusion) generate vectors directly from the product's image and title at listing time.
  * The new product is instantly projected into the semantic neighborhood of similar established products.
  * As a result, the new product immediately appears in recommendations on day one without requiring a single historical purchase."""

q4 = """4. How does Reverse Image Search handle changes in image resolution, cropping, and lighting?"""
a4 = """* Traditional pixel matching or image hashing (pHash) breaks when an image is cropped, re-compressed, or shot under different lighting.
* Modern Vision Transformers (ViT, CLIP, DINOv2) are trained on self-supervised contrastive objectives with heavy data augmentations (random cropping, color jitter, affine transformations).
* The models learn invariant high-level representations (edges, textures, spatial object relationships) rather than raw RGB pixel values.
* Consequently, a user photo of a shoe taken in poor fluorescent lighting matches the studio white-background product catalog photo with high cosine similarity."""

q5 = """5. How are Customer Behavior Profiles embedded into vectors for lookalike audience targeting?"""
a5 = """* **Graph Embeddings (Node2Vec / GraphSAGE)**: User transaction networks (users connecting to merchant nodes) are embedded such that users with similar financial transaction patterns occupy proximate coordinates.
* **Sequential Embeddings (Word2Vec on Event Streams)**: A customer's sequence of website interactions (view_page -> add_to_cart -> abandon) is treated like a sentence of tokens and encoded via transformer encoders.
* **Lookalike Clustering**: Marketing teams select a "seed cohort" of 500 high-LTV customers, compute their centroid vector, and query the vector database for the top 50,000 nearest neighbor user profiles."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Practical Similarity Search Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The Asymmetric Question-Answer Fallacy**  
> Indexing only answers (articles) in a customer support vector database and querying them with raw user questions 
> often fails because articles describe technical solutions using formal jargon, while users express problems using 
> emotional symptoms ("my screen went totally black!"). Index previous user queries alongside articles to enable Q2Q matching.

> [!WARNING]
> **Trap 2: Aspect Ratio Distortion in Vision Encoders**  
> Naively resizing rectangular clothing photos (e.g. 1080x1920) into square 224x224 inputs without letterboxing 
> skews object silhouettes, causing tall dresses to look like shirts. Always preserve aspect ratio with padding prior 
> to vision transformer inference.

> [!WARNING]
> **Trap 3: Stale Customer Behavioral Embeddings**  
> Customer behavior evolves rapidly. A user who was shopping for baby strollers 6 months ago may now be interested in toddlers' toys. 
> Batch pipelines that recompute customer embeddings only once a month create stale recommendations. 
> Implement time-decay weighting on event history.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What is Q2Q vs Q2A?"):
        st.markdown("""
        **Q2Q vs Q2A**:  
        Question-to-Question matching pairs symmetric user inquiries for FAQ deflection; Question-to-Answer retrieves asymmetric explanatory passages.
        """)
    with st.expander("🗂️ Flashcard: What is a Two-Tower Model?"):
        st.markdown("""
        **Two-Tower Architecture**:  
        Decoupled neural networks where User and Item towers output vectors into a shared space, enabling sub-10ms candidate retrieval over 100M+ items.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: How Does Visual Search Handle Lighting?"):
        st.markdown("""
        **Vision Transformer Invariance**:  
        Models like CLIP and DINOv2 learn high-level semantic features invariant to image resolution, lighting shifts, and cropping.
        """)
    with st.expander("🗂️ Flashcard: What is a Lookalike Audience?"):
        st.markdown("""
        **Lookalike Modeling**:  
        Querying a vector database using the centroid vector of high-value customers to discover the nearest unengaged user profiles with matching behaviors.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I can articulate the system design for all 5 enterprise similarity search domains.", value=True)
chk2 = st.checkbox("I understand why Question-to-Question matching outperforms Question-to-Answer matching for ticket deflection.")
chk3 = st.checkbox("I know how Two-Tower recommendation architectures decouple offline item indexing from online query inference.")
chk4 = st.checkbox("I understand how content-based embeddings resolve the e-commerce cold-start problem.")
chk5 = st.checkbox("I know how to represent user behavioral event sequences as vectors for lookalike clustering.")
chk6 = st.checkbox("I can implement a multi-domain vector search engine in Python.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Incredible achievement! You have mastered Practical Examples and completed Chapter 3 of the Vector Database curriculum!")
