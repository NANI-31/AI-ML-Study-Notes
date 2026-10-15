import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="28.2 Content-Based Recommendations - Embeddings Curriculum",
    page_icon="📦",
    layout="wide"
)

# Title & Overview
st.title("28.2 Content-Based Recommendations: Products, Articles, Movies & Job Embeddings")
st.caption("Chapter 28: Embeddings for Recommendations | Domain: 3. Vector Embeddings")
st.markdown("""
While collaborative filtering recommends items based on *what other users with similar tastes liked*, 
**Content-Based Recommendation** recommends items based on the **intrinsic features and semantic properties of the items themselves**.

In modern industry platforms, unstructured text and metadata are serialized and passed into dense embedding models to generate:
* **Product Embeddings (E-Commerce)**: Capturing brand, specifications, design aesthetic, price point, and product descriptions (e.g., Amazon, Shopify).
* **Article Embeddings (News & Media)**: Capturing editorial topics, narrative tone, journalistic style, and subject matter (e.g., Medium, Substack, Bloomberg).
* **Movie Embeddings (Streaming Video)**: Capturing plot synopses, genre combinations, cinematic tropes, and character dynamics (e.g., Netflix, IMDb).
* **Job Embeddings (Talent Matching)**: Capturing required skills, technical stacks, seniority levels, and responsibilities to match candidate resumes with open requisitions (e.g., LinkedIn, Indeed).

Because content-based embeddings depend solely on item metadata, they **completely bypass the item cold-start problem**, allowing 
newly published products, articles, movies, or job postings to be discovered and recommended immediately.
""")

st.divider()

# Section 1: Content Serialization Strategies Across Domains
st.header("1. Metadata Serialization Strategies Across Four Core Domains", divider="rainbow")

st.markdown("""
To create high-quality content embeddings, raw database records must be serialized into structured, descriptive text blocks 
that transformer embedding models can digest effectively:
""")

col_ser1, col_ser2 = col_ser1, col_ser2 = st.columns([1, 1])

with col_ser1:
    st.markdown("""
    ### 1. Product Serialization (E-Commerce)
    * **Key Attributes**: Brand, Title, Category Hierarchy, Key Specs, Material, Target Audience.
    * **Serialized Format**:
      ```text
      Product: Sony WH-1000XM5 Wireless Headphones
      Brand: Sony | Category: Electronics > Audio > Over-Ear Headphones
      Features: Industry-leading active noise cancellation, 30-hour battery life, 
      multipoint Bluetooth pairing, lightweight carbon fiber headband, premium memory foam.
      Price Tier: Premium ($350 - $400)
      ```
    
    ### 2. Article Serialization (News & Media)
    * **Key Attributes**: Headline, Subtitle, Author, Tags/Taxonomy, Leading Paragraphs.
    * **Serialized Format**:
      ```text
      Article: Understanding Kubernetes Pod Autoscaling Under Extreme Traffic Spikes
      Author: Cloud Engineering Team | Tags: DevOps, Kubernetes, HPA, SRE
      Summary: In-depth technical breakdown of Horizontal Pod Autoscaler metrics latency, 
      Prometheus Adapter configuration, and tuning stabilization windows to prevent 
      thrashing during sudden traffic surges.
      ```
    """)

with col_ser2:
    st.markdown("""
    ### 3. Movie Serialization (Streaming)
    * **Key Attributes**: Title, Year, Director, Genres, Cast, Plot Synopsis, Mood/Tone.
    * **Serialized Format**:
      ```text
      Movie: Interstellar (2014)
      Director: Christopher Nolan | Genres: Sci-Fi, Adventure, Drama
      Themes: Space exploration, black holes, time dilation, survival of human species.
      Synopsis: When Earth becomes uninhabitable, a team of astronauts travels through 
      a newly discovered wormhole near Saturn in search of a habitable exoplanet for humanity.
      ```
    
    ### 4. Job Serialization (Talent Matching)
    * **Key Attributes**: Job Title, Seniority, Required Skills, Tech Stack, Compensation, Location.
    * **Serialized Format**:
      ```text
      Role: Senior Backend Systems Engineer
      Level: Senior (5+ years) | Location: Remote (US)
      Tech Stack: Go, PostgreSQL, Kafka, Kubernetes, Distributed Systems, High-Concurrency.
      Responsibilities: Architect low-latency event-driven microservices handling 
      100,000+ QPS, optimize database connection pooling, and maintain high availability.
      ```
    """)

st.divider()

# Section 2: End-to-End Content-Based Recommendation Architecture
st.header("2. Architecture of Content-Based Recommenders", divider="rainbow")

col_arch1, col_arch2 = col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    st.markdown("""
    ### The "More Like This" Workflow
    1. **Catalog Indexing (Offline Batch)**:
       * All items in the database are serialized into canonical text descriptions.
       * A dense sentence transformer (e.g., `BAAI/bge-large-en-v1.5` or `OpenAI text-embedding-3-small`) encodes each description into an L2-normalized vector.
       * Vectors are stored in an HNSW vector database index alongside metadata payloads.
    2. **Item-to-Item Similarity (Real-Time Search)**:
       * When a user views an item (e.g., Product A or Movie X), its pre-computed vector `v_A` is passed directly as the query vector.
       * HNSW finds the top-k nearest neighbors in sub-5ms:
         `Sim(A, B) = dot(v_A, v_B)`
    3. **Cold-Start Advantages**:
       * A new movie added to the streaming platform 10 seconds ago can be recommended immediately on similar movie pages because its synopsis and genre vector already position it in the correct visual and thematic cluster!
    """)

with col_arch2:
    st.markdown("### Content-Based Vector Matching Pipeline")
    cb_dot = """
    digraph ContentBasedPipeline {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        ActiveItem [label="User Currently Viewing Item\\n'Interstellar (2014)'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        RetrieveVec [label="Look up Pre-computed Vector\\nv_target in R^1024", fillcolor="#DBEAFE"];

        VectorStore [label="HNSW Catalog Index\\n1,000,000 Movie Embeddings\\n(Stored in Qdrant / Milvus)", fillcolor="#E0E7FF", color="#3B82F6", style="filled,bold"];

        CosineANN [label="Cosine ANN Search\\nFinds Nearest Neighbors in 3ms", fillcolor="#FEF3C7", color="#F59E0B"];

        TopRecs [label="Recommended 'More Like This'\\n1. The Martian (Sim: 0.92)\\n2. Contact (Sim: 0.89)\\n3. Arrival (Sim: 0.87)", fillcolor="#D1FAE5", color="#10B981", style="filled,bold"];

        ActiveItem -> RetrieveVec;
        RetrieveVec -> CosineANN;
        VectorStore -> CosineANN;
        CosineANN -> TopRecs;
    }
    """
    st.graphviz_chart(cb_dot, use_container_width=False)

st.divider()

# Section 3: Diversity, Novelty & Breaking the Filter Bubble
st.header("3. Overcoming the Filter Bubble: Diversity & Novelty Injection", divider="rainbow")

col_div1, col_div2 = col_div1, col_div2 = st.columns([1, 1])

with col_div1:
    st.markdown("""
    ### The Content-Based Filter Bubble
    A notorious weakness of naive content-based systems is **overspecialization**:
    * If a user reads an article about *Rust Memory Safety*, a pure cosine search will recommend 10 more articles about *Rust Memory Safety*.
    * The user gets trapped in a repetitive "echo chamber" with zero serendipity or exposure to related topics (e.g., *Go Concurrency* or *Zig Systems Programming*).

    ### Production Remediation: Maximal Marginal Relevance (MMR)
    To balance relevance with variety, production systems apply **MMR Re-ranking**:
    
    ```text
    MMR_Score(d) = lambda * Sim(d, query) - (1 - lambda) * Max_{s in Selected} [ Sim(d, s) ]
    ```
    
    Where:
    * `lambda` (e.g., 0.65) controls the trade-off: higher `lambda` favors pure relevance; lower `lambda` forces diverse subtopics.
    * The second term penalizes candidates that are too similar to items already picked in the recommendation list!
    """)

with col_div2:
    st.markdown("### Production Code: Content Recommender with MMR")
    st.code("""
import numpy as np

def maximal_marginal_relevance(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    candidate_ids: list[str],
    top_k: int = 5,
    lambda_param: float = 0.65
) -> list[str]:
    \"\"\"Selects top-k candidates balancing similarity and diversity.\"\"\"
    selected_indices = []
    remaining_indices = list(range(len(candidate_ids)))
    
    # 1. First item is the highest cosine match
    base_sims = np.dot(candidate_vecs, query_vec)
    best_first = int(np.argmax(base_sims))
    selected_indices.append(best_first)
    remaining_indices.remove(best_first)
    
    # 2. Greedily pick remaining items using MMR formula
    while len(selected_indices) < top_k and remaining_indices:
        best_score = -float('inf')
        best_idx = None
        
        for idx in remaining_indices:
            sim_to_query = base_sims[idx]
            max_sim_to_selected = max([
                np.dot(candidate_vecs[idx], candidate_vecs[s])
                for s in selected_indices
            ])
            
            mmr = lambda_param * sim_to_query - (1 - lambda_param) * max_sim_to_selected
            if mmr > best_score:
                best_score = mmr
                best_idx = idx
                
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
    return [candidate_ids[i] for i in selected_indices]
""", language="python")

st.divider()

# Section 4: Interactive Workbench — 4-Domain Content Recommender Sandbox
st.header("4. Interactive Workbench: Multi-Domain Content Recommender Simulator", divider="rainbow")

st.markdown("""
Test content-based recommendations live across four real-world domains: 
**E-Commerce Products**, **Tech Articles**, **Movies**, and **Job Listings**.
Choose an active item to inspect its metadata and see the highest-similarity recommendations generated via vector distance!
""")

domain_catalogs = {
    "E-Commerce Products": {
        "items": [
            {"id": "PROD_01", "title": "Ergonomic Mechanical Keyboard (Keychron K2)", "category": "Keyboards", "brand": "Keychron", "desc": "Wireless mechanical keyboard with Gateron brown switches, RGB backlighting, Mac and Windows compatibility, compact 75% layout.", "tags": ["keyboard", "mechanical", "wireless", "ergonomic", "bluetooth"]},
            {"id": "PROD_02", "title": "Custom Ergonomic Split Keyboard (Ergodox EZ)", "category": "Keyboards", "brand": "Ergodox", "desc": "Ortholinear split mechanical keyboard with hot-swappable switches, dual USB cables, tilt kit, and programmable layers for wrist strain relief.", "tags": ["keyboard", "mechanical", "ergonomic", "split", "programmable"]},
            {"id": "PROD_03", "title": "Vertical Ergonomic Wireless Mouse (Logitech MX Vertical)", "category": "Mice", "brand": "Logitech", "desc": "Ergonomic vertical mouse with 57-degree handshake grip angle, optical sensor, rechargeable USB-C battery, and multi-device flow.", "tags": ["mouse", "ergonomic", "vertical", "wireless", "wrist pain"]},
            {"id": "PROD_04", "title": "4K Ultra-Wide Curved Monitor 34-inch (Dell U3423WE)", "category": "Monitors", "brand": "Dell", "desc": "WQHD 34-inch curved IPS monitor with integrated KVM switch, 90W USB-C power delivery, and built-in speakers for productivity.", "tags": ["monitor", "curved", "ultrawide", "usb-c", "display"]},
            {"id": "PROD_05", "title": "Ultra-Lightweight Wireless Gaming Mouse (Logitech G Pro X)", "category": "Mice", "brand": "Logitech", "desc": "Superlight esports gaming mouse weighing 63 grams with Hero 25K optical sensor and zero-additive PTFE feet.", "tags": ["mouse", "gaming", "wireless", "lightweight", "esports"]}
        ]
    },
    "Tech Articles": {
        "items": [
            {"id": "ART_01", "title": "Deep Dive: PostgreSQL Indexing & B-Trees", "category": "Database Engineering", "brand": "Engineering Blog", "desc": "Comprehensive analysis of PostgreSQL storage layout, WAL buffers, B-Tree index page splitting, and indexing strategies for high-write tables.", "tags": ["postgres", "database", "indexes", "b-tree", "sql"]},
            {"id": "ART_02", "title": "Optimizing Query Performance with Postgres EXPLAIN ANALYZE", "category": "Database Engineering", "brand": "SRE Weekly", "desc": "A practical guide to reading query execution plans, spotting sequential scans, tuning work_mem, and preventing lock contention in PostgreSQL.", "tags": ["postgres", "database", "performance", "explain", "optimization"]},
            {"id": "ART_03", "title": "Distributed Caching with Redis Cluster & Twemproxy", "category": "Distributed Systems", "brand": "System Design", "desc": "Architectural guide to scaling in-memory caching across multi-node Redis clusters with consistent hashing, replication, and failover.", "tags": ["redis", "caching", "distributed systems", "clusters", "in-memory"]},
            {"id": "ART_04", "title": "Zero-Downtime Database Migrations with Flyway", "category": "Database Engineering", "brand": "DevOps Daily", "desc": "How to execute backward-compatible schema changes, column additions, and view refactoring in production without table locks.", "tags": ["database", "migrations", "devops", "zero downtime", "sql"]},
            {"id": "ART_05", "title": "Mastering Docker Multi-Stage Builds for Tiny Images", "category": "Cloud & Containers", "brand": "Cloud Native", "desc": "Techniques for pruning build tools, minimizing vulnerability attack surfaces, and speeding up CI/CD pipeline artifact delivery.", "tags": ["docker", "containers", "devops", "ci-cd", "optimization"]}
        ]
    },
    "Movie Streaming": {
        "items": [
            {"id": "MOV_01", "title": "Interstellar (2014)", "category": "Sci-Fi / Space", "brand": "Christopher Nolan", "desc": "Astronauts travel through a mysterious wormhole near Saturn in search of a new habitable planet to save humanity from global crop blight.", "tags": ["space", "wormhole", "sci-fi", "time dilation", "gravity", "nolan"]},
            {"id": "MOV_02", "title": "The Martian (2015)", "category": "Sci-Fi / Space", "brand": "Ridley Scott", "desc": "An astronaut becomes stranded on Mars after a fierce storm and must use his botanical engineering ingenuity to survive until a rescue mission arrives.", "tags": ["mars", "space", "astronaut", "survival", "science", "botany"]},
            {"id": "MOV_03", "title": "Arrival (2016)", "category": "Sci-Fi / First Contact", "brand": "Denis Villeneuve", "desc": "A linguistic professor is recruited by the military to communicate with enigmatic extraterrestrial visitors who land giant monolith ships across Earth.", "tags": ["aliens", "linguistics", "sci-fi", "first contact", "time", "villeneuve"]},
            {"id": "MOV_04", "title": "Inception (2010)", "category": "Sci-Fi / Heist", "brand": "Christopher Nolan", "desc": "A skilled corporate thief who steals secrets through dream-sharing technology is offered a chance to erase his criminal past by planting an idea.", "tags": ["dreams", "heist", "mind", "nolan", "subconscious", "reality"]},
            {"id": "MOV_05", "title": "The Grand Budapest Hotel (2014)", "category": "Comedy / Drama", "brand": "Wes Anderson", "desc": "The whimsical adventures of a legendary concierge at a famous European hotel between the wars and his friendship with a trusted lobby boy.", "tags": ["hotel", "quirky", "comedy", "wes anderson", "european", "friendship"]}
        ]
    },
    "Job Listings": {
        "items": [
            {"id": "JOB_01", "title": "Senior Machine Learning Engineer (LLMs & RAG)", "category": "Artificial Intelligence", "brand": "AI Labs Corp", "desc": "Build scalable Retrieval-Augmented Generation pipelines, fine-tune transformer models, implement vector databases (Milvus/Qdrant), and optimize GPU inference.", "tags": ["python", "pytorch", "transformers", "rag", "vector db", "llm"]},
            {"id": "JOB_02", "title": "Staff AI Search & Retrieval Engineer", "category": "Artificial Intelligence", "brand": "Global Cloud", "desc": "Architect enterprise semantic search engines using BGE embeddings, hybrid BM25 search, cross-encoder rerankers, and large-scale ANN indexing.", "tags": ["search", "embeddings", "information retrieval", "python", "nlp", "reranking"]},
            {"id": "JOB_03", "title": "Distributed Systems Engineer (Go & Kafka)", "category": "Infrastructure", "brand": "Fintech Platform", "desc": "Design high-throughput event streaming microservices in Go, optimize Kafka broker partitions, and maintain 99.999% availability.", "tags": ["golang", "kafka", "distributed systems", "microservices", "concurrency"]},
            {"id": "JOB_04", "title": "Frontend React / Next.js Architect", "category": "Frontend Engineering", "brand": "SaaS Studio", "desc": "Lead architecture for enterprise web applications using React 19, Next.js Server Components, Tailwind CSS, and web accessibility standards.", "tags": ["react", "nextjs", "typescript", "frontend", "tailwind", "css"]},
            {"id": "JOB_05", "title": "Machine Learning Platform Engineer (MLOps)", "category": "Artificial Intelligence", "brand": "Enterprise Tech", "desc": "Deploy Kubernetes Kubeflow clusters, configure Triton inference servers, automate model CI/CD pipelines, and monitor model drift.", "tags": ["mlops", "kubernetes", "triton", "docker", "python", "ci-cd"]}
        ]
    }
}

selected_domain = st.selectbox("Select Recommendation Domain:", options=list(domain_catalogs.keys()))
active_catalog = domain_catalogs[selected_domain]["items"]

col_sb1, col_sb2 = col_sb1, col_sb2 = st.columns([1, 1])

with col_sb1:
    st.subheader("Step 1: Choose Active / Viewed Item")
    item_titles = {f"{it['id']} - {it['title']}": it for it in active_catalog}
    selected_item_key = st.selectbox("Current Item in View:", options=list(item_titles.keys()))
    active_item = item_titles[selected_item_key]
    
    st.info(f"""
    **Item**: `{active_item['title']}`  
    **Category**: `{active_item['category']}` | **Entity / Creator**: `{active_item['brand']}`  
    **Serialized Description**: *"{active_item['desc']}"*  
    **Tags**: `{'`, `'.join(active_item['tags'])}`
    """)

# Simulated Cosine Similarity Engine
def simulate_content_recommendations(target_item, catalog):
    results = []
    target_tags = set(target_item["tags"])
    
    for it in catalog:
        if it["id"] == target_item["id"]:
            continue
            
        it_tags = set(it["tags"])
        common_tags = target_tags.intersection(it_tags)
        
        # Calculate realistic similarity score based on category and tag overlap
        category_match = (it["category"] == target_item["category"])
        brand_match = (it["brand"] == target_item["brand"])
        
        base_sim = 0.40
        if category_match:
            base_sim += 0.28
        if brand_match:
            base_sim += 0.12
            
        tag_boost = len(common_tags) * 0.08
        sim = min(0.965, round(base_sim + tag_boost + np.random.uniform(0.01, 0.04), 3))
        
        results.append({
            "Item ID": it["id"],
            "Recommended Item": it["title"],
            "Category": it["category"],
            "Shared Tags": ", ".join(list(common_tags)) if common_tags else "None",
            "Cosine Similarity": sim
        })
    return pd.DataFrame(results).sort_values(by="Cosine Similarity", ascending=False).reset_index(drop=True)

with col_sb2:
    st.subheader("Step 2: Recommended 'More Like This' Items")
    df_recs = simulate_content_recommendations(active_item, active_catalog)
    
    styled_recs = df_recs.style.format({"Cosine Similarity": "{:.3f}"}).background_gradient(
        subset=["Cosine Similarity"], cmap="YlGn", vmin=0.4, vmax=1.0
    )
    st.dataframe(styled_recs, use_container_width=True)
    
    top_pick = df_recs.iloc[0]
    st.success(f"""
    **Top Content-Based Recommendation**: `{top_pick['Recommended Item']}` *(Similarity: **{top_pick['Cosine Similarity']:.3f}**)*  
    *Shared Content Attributes*: `{top_pick['Shared Tags']}`
    """)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: What are the fundamental differences between Collaborative Filtering and Content-Based Recommendation?", expanded=True):
    st.markdown("""
    **Answer:**
    * **Collaborative Filtering (CF)**:
      * Recommends items based on **historical user behavior patterns** (e.g., clicks, ratings, purchases).
      * *Core intuition*: If User A and User B both liked Item X and Item Y, User A will probably like Item Z that User B enjoyed.
      * *Weakness*: Severe cold-start problem (cannot recommend brand new items with 0 clicks); struggles with sparse user matrices.
    * **Content-Based Filtering (CB)**:
      * Recommends items based on the **intrinsic descriptive attributes of the items** (text descriptions, specifications, tags, visual embeddings).
      * *Core intuition*: If User A liked Item X (a sci-fi movie with time dilation), recommend Item Y (another sci-fi movie with time dilation).
      * *Strengths*: Zero cold-start for new items; highly transparent and explainable recommendations; operates without needing millions of other users' data.
      * *Weakness*: Risk of overspecialization (the "Filter Bubble") and inability to capture subjective quality (e.g., a poorly written movie with the same tags as a masterpiece).
    """)

with st.expander("Q2: How do you serialize complex multi-attribute items (such as jobs or e-commerce products) into a single text embedding?"):
    st.markdown("""
    **Answer:**
    To convert tabular or JSON item records into high-density text embeddings:
    1. **Hierarchical Attribute Prefixing**: Prefix key fields with their semantic attribute names:
       `Title: {title} | Category: {category} | Brand: {brand} | Specs: {specs}`
    2. **Domain-Specific Term Normalization**: Normalize acronyms and taxonomy levels (e.g., convert `NYC` to `New York City, New York` and `L6` to `Staff Software Engineer (Senior)`).
    3. **Order by Importance**: Place the most discriminative signals (Title, Brand, Core Technologies) at the beginning of the text string to ensure they are captured even if sequence truncation occurs.
    4. **Weighted Pooling or Multi-Vector Representation**: In advanced pipelines, embed Title, Description, and Tags separately, and compute a learned weighted average:
       `v_item = w1 * v_title + w2 * v_desc + w3 * v_tags`
    """)

with st.expander("Q3: How does Maximal Marginal Relevance (MMR) prevent overspecialization in content recommenders?"):
    st.markdown("""
    **Answer:**
    Naive content-based retrieval ranks items strictly by cosine similarity to the target item:
    `Score(c) = Sim(c, target)`
    This causes top results to be near-identical duplicates (e.g., 5 identical versions of the same charging cable).
    
    **MMR (Carbonell & Goldstein)** solves this by optimizing a combined objective that penalizes redundancy:
    `MMR(c) = lambda * Sim(c, target) - (1 - lambda) * Max_{s in Selected} [ Sim(c, s) ]`
    * The first term rewards similarity to the user's active item.
    * The second term penalizes candidates that are too similar to items already selected in the recommendation list.
    * Setting `lambda = 0.65` guarantees that each subsequent recommendation covers a distinct sub-genre, brand, or feature dimension while retaining high overall relevance.
    """)

with st.expander("Q4: Can content-based embeddings be combined with collaborative interaction signals?"):
    st.markdown("""
    **Answer:**
    Yes, this is the foundation of modern **Hybrid Recommender Systems**:
    * **Two-Tower Neural Networks**: In models like YouTube DNN, the Item Tower takes both **content features** (video title embedding, tags) AND **collaborative features** (historical co-watch frequency, channel ID embeddings).
    * **Vector Concatenation / Fusion**: Combine a 256-d content embedding with a 128-d graph collaborative embedding (Node2Vec or Graph Convolutional Network - PinSage):
      `v_combined = W * [v_content; v_graph]`
    This delivers the best of both worlds: zero-shot cold-start discovery for new items combined with the predictive power of crowd wisdom for mature items.
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Concatenating Raw JSON Directly Into Embedding Models**: Feeding `{"id": 104, "cat_id": 992, "ts": 1728392}` into a text transformer. Numerical database IDs produce random meaningless token embeddings. Always convert fields into human-readable semantic prose.
* **Trap 2: Ignoring Text Token Truncation on Long Descriptions**: Passing a 2,000-word product description into a model with a 512-token context window without placing key attributes at the beginning. The critical product specifications at the end get silently cut off.
* **Trap 3: Static Pre-computing Without Cache Invalidation**: Pre-computing item vectors once during product creation and never updating them when a seller updates the product title, specs, or compatibility tags.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: Content vs. CF"):
        st.markdown("**Front**: What is the main advantage of Content-Based over Collaborative Filtering?")
        st.divider()
        st.markdown("**Back**: **Zero Cold-Start**. Content-based systems can immediately recommend brand new items with zero historical user clicks.")

with card_cols[1]:
    with st.popover("Card 2: Filter Bubble"):
        st.markdown("**Front**: What is the Filter Bubble in content recommenders?")
        st.divider()
        st.markdown("**Back**: Overspecialization where the user is only shown items identical to what they just viewed, with zero diversity or discovery.")

with card_cols[2]:
    with st.popover("Card 3: MMR"):
        st.markdown("**Front**: What does MMR do in recommendations?")
        st.divider()
        st.markdown("**Back**: Maximal Marginal Relevance explicitly balances relevance to the target item with diversity among selected recommendations.")

with card_cols[3]:
    with st.popover("Card 4: Serialization"):
        st.markdown("**Front**: Why is metadata serialization important for content embeddings?")
        st.divider()
        st.markdown("**Back**: It formats discrete database fields into structured, descriptive natural language sentences that language models can meaningfully encode.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Content-Based Recommendations:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand content-based recommendation across Products, Articles, Movies, and Jobs.", key="chk28_2_1")
    st.checkbox("I know how metadata serialization turns database fields into dense vectors.", key="chk28_2_2")
    st.checkbox("I understand why content-based embeddings solve the item cold-start problem.", key="chk28_2_3")
with c2:
    st.checkbox("I can implement Maximal Marginal Relevance (MMR) to break filter bubbles.", key="chk28_2_4")
    st.checkbox("I understand how hybrid recommenders combine content with collaborative signals.", key="chk28_2_5")
    st.checkbox("I know the failure modes of raw JSON concatenation and token truncation.", key="chk28_2_6")
