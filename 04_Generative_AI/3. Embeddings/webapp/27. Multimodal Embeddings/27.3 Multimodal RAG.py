import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="27.3 Multimodal RAG - Embeddings Curriculum",
    page_icon="📑",
    layout="wide"
)

# Title & Overview
st.title("27.3 Multimodal RAG: Images, PDFs, Tables, Text & Combined Visual-Textual Retrieval")
st.caption("Chapter 27: Multimodal Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
In real-world enterprise environments, documents are rarely clean, plain-text Markdown files. 
The most critical business knowledge—financial statements, technical manuals, architectural blueprints, medical records, and legal filings—is 
locked inside **complex, visually rich PDFs**:
* **High-density Tables**: Multi-column balance sheets, row spans, and numerical matrices where standard linear text chunking destroys row-column relationships.
* **Information-Dense Visuals**: Flowcharts, system architecture diagrams, and bar charts that contain zero raw text characters.
* **Complex Multi-Column Layouts**: Magazine-style layouts where naive OCR reads horizontally across columns, interweaving unrelated paragraphs into incoherent gibberish.

**Multimodal RAG** transcends the limitations of traditional text-only pipelines by indexing, retrieving, and synthesizing 
across **Text, Tables, Images, and Page Layouts** simultaneously.
""")

st.divider()

# Section 1: The Three Multimodal RAG Architectures
st.header("1. Architectural Taxonomy of Multimodal RAG Systems", divider="rainbow")

col_tax1, col_tax2 = col_tax1, col_tax2 = st.columns([1, 1])

with col_tax1:
    st.markdown("""
    ### The 3 Core Multimodal RAG Architectures
    
    1. **Pipeline 1: Text-Conversion Pipeline (OCR + VLM Captioning)**:
       * *Ingestion*: Optical Character Recognition (Tesseract, PaddleOCR) extracts prose. A Vision-Language Model (e.g., GPT-4o-mini) inspects images/charts and generates textual summaries. Tables are parsed to Markdown.
       * *Retrieval*: Standard text embedding model (e.g., `text-embedding-3`).
       * *Pros*: Works with existing text vector stores and legacy infrastructure.
       * *Cons*: Brittle OCR layout errors; loss of subtle visual graphics and spatial chart positions.
       
    2. **Pipeline 2: Multi-Vector Dual-Store (Multi-Index Fusion)**:
       * *Ingestion*: Text chunks are stored in a text vector collection (BGE/OpenAI); extracted image/chart crops are stored in a CLIP/SigLIP vector collection with shared Document ID foreign keys.
       * *Retrieval*: Query searches both indices; results are merged via Reciprocal Rank Fusion (RRF).
       * *Pros*: Native visual vector matching; retrieves exact diagram crops.
       * *Cons*: Two separate embedding models, two indices to sync and maintain.
       
    3. **Pipeline 3: Vision-Native Document Retrieval (ColPali)** *(State of the Art)*:
       * *Ingestion*: Treat every PDF page as an **image screenshot**. Feed the page image into a Vision-Language foundation model (PaliGemma/SigLIP) with **ColBERT late-interaction patch tokens**.
       * *Retrieval*: Query is text tokens; PDF is image patch tokens. Computes MaxSim late interaction directly on the visual page!
       * *Pros*: Zero OCR required! Flawlessly understands complex charts, tables, formulas, and layouts.
    """)

with col_tax2:
    st.markdown("### ColPali: Vision-Native Page Retrieval Architecture")
    colpali_dot = """
    digraph ColPaliArchitecture {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        PDFPage [label="Enterprise PDF Page\\nRendered as Image (1024x1024)\\n(Contains text, chart, and table)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        PaliGemma [label="Vision-Language Encoder (PaliGemma)\\nGenerates Multi-Vector Patch Embeddings\\n(1024 patch vectors per page)", fillcolor="#DBEAFE"];

        QueryTxt [label="User Text Query\\n'What was Q3 cloud revenue growth?'", fillcolor="#FEF3C7", color="#F59E0B", style="filled,bold"];

        TxtEnc [label="ColPali Text Encoder\\nGenerates token vectors q_1..q_m", fillcolor="#FDE68A"];

        LateInteract [label="ColBERT Late Interaction (MaxSim)\\nEvery text token attends to\\nnearest visual page patch!", fillcolor="#D1FAE5", color="#10B981", style="filled,bold"];

        MultimodalLLM [label="Multimodal LLM Synthesis (GPT-4o)\\nReceives retrieved PDF image page crop\\nGenerates grounded answer with table facts", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        PDFPage -> PaliGemma;
        QueryTxt -> TxtEnc;
        PaliGemma -> LateInteract;
        TxtEnc -> LateInteract;
        LateInteract -> MultimodalLLM;
    }
    """
    st.graphviz_chart(colpali_dot, use_container_width=False)

st.markdown("""
### Architecture Trade-Off Comparison
| Capability / Metric | Pipeline 1: OCR + VLM Captioning | Pipeline 2: Dual-Store (Text + CLIP) | Pipeline 3: ColPali (Vision-Native) |
| :--- | :--- | :--- | :--- |
| **Ingestion Complexity** | High (Multi-step OCR + LLM calls) | Moderate (Parser + Image Cropper) | **Lowest (Render PDF page to image)** |
| **Table Preservation** | Moderate (Markdown tables break on spans) | Weak (Tables converted to prose) | **Flawless (Reads raw visual table layout)** |
| **Infographic & Diagram Understanding** | Limited to LLM caption quality | Coarse (CLIP 77-token ceiling) | **Pixel-Level (Attends to visual graph lines)** |
| **Storage / Memory Footprint** | Low (Text strings only) | Moderate (Text + 512-d CLIP vectors) | High (Multi-vector: ~1024 vectors per page) |
| **Retrieval Accuracy (Complex Layouts)** | 62% - 74% | 70% - 81% | **89% - 96%** |
""")

st.divider()

# Section 2: Table Representation Strategies for Vector Retrieval
st.header("2. Representing Complex Tables in Multimodal RAG", divider="rainbow")

st.markdown("""
Tables are among the most difficult modalities in RAG. 
Consider a financial table where Row 1 is *"Operating Expenses"* and Column 4 is *"Q3 2024 Actual ($M)"*.
If a standard text chunker splits the table at character 500:
* The column headers are detached from the row data!
* A vector search for *"Q3 2024 operating expenses"* retrieves numbers with zero column labels attached.
""")

col_tbl1, col_tbl2 = col_tbl1, col_tbl2 = st.columns([1, 1])

with col_tbl1:
    st.markdown("""
    ### The 4 Table Serialization Formats
    1. **Markdown Format**:
       * Clean, human-readable, and compact.
       * *Best for*: Simple 2D tables without row-spans or merged cells.
    2. **HTML `<table>` Format**:
       * Preserves semantic attributes like `rowspan="2"`, `colspan="3"`, and nested headers.
       * *Best for*: Complex corporate SEC 10-K filings and clinical trial data.
    3. **Key-Value Deconstruction (Row Linearization)**:
       * Converts each table row into an independent declarative sentence:
         * *Example*: *"In Q3 2024, Cloud Segment Operating Expenses were $450 Million, representing a 14% year-over-year increase."*
       * *Best for*: Dense vector similarity search where sentences match query semantics.
    4. **Dual Representation (Markdown + VLM Summary)**:
       * Store the raw HTML/Markdown table in document metadata for prompt injection.
       * Compute embeddings over an LLM-generated comprehensive table summary.
    """)

with col_tbl2:
    st.markdown("### Production Table Serialization Code")
    st.code("""
def linearize_table_row(headers: list[str], row: list[str], doc_title: str) -> str:
    \"\"\"Converts a table row into an unambiguous declarative sentence.\"\"\"
    pairs = [f"{h}: {val}" for h, val in zip(headers, row) if val.strip()]
    return f"From {doc_title} Table - " + ", ".join(pairs)

# Example Table Row:
# Headers: ['Metric', 'Q1 2024', 'Q2 2024', 'Q3 2024']
# Values:  ['Net Profit Margin', '14.2%', '16.5%', '18.1%']
# Linearized: 
# "From Financial Report 2024 Table - Metric: Net Profit Margin, Q1 2024: 14.2%, Q2 2024: 16.5%, Q3 2024: 18.1%"
""", language="python")

st.divider()

# Section 3: Grounded Multimodal Synthesis with Vision-Language Models
st.header("3. Grounded Multimodal Synthesis with Frontier Vision LLMs", divider="rainbow")

st.markdown("""
Once relevant multimodal chunks (PDF page screenshots, cropped diagrams, table Markdown) are retrieved, 
they are packaged into a multimodal prompt for a Vision-Language Model (e.g., **GPT-4o**, **Claude 3.5 Sonnet**, or **Gemini 1.5 Pro**).
""")

col_syn1, col_syn2 = col_syn1, col_syn2 = st.columns([1, 1])

with col_syn1:
    st.markdown("### Multimodal API Payload Structure (OpenAI GPT-4o)")
    st.code("""
import base64
from openai import OpenAI

client = OpenAI()

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def synthesize_multimodal_answer(query: str, retrieved_images: list[str], table_md: str):
    image_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image_base64(img)}"}
        }
        for img in retrieved_images
    ]
    
    messages = [
        {
            "role": "system",
            "content": "You are a multimodal technical assistant. Answer the user query using the attached PDF page images and tabular data. Ground all claims with specific visual bounding references."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Question: {query}\\n\\nRetrieved Table Data:\\n{table_md}"},
                *image_content
            ]
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500
    )
    return response.choices[0].message.content
""", language="python")

with col_syn2:
    st.markdown("### Visual Grounding & Citation Strategy")
    st.markdown("""
    To ensure auditability and zero hallucination in regulated industries (healthcare, finance, aerospace):
    * **Visual Bounding Boxes**: Prompt the model to cite specific quadrants or bounding coordinates `[x1, y1, x2, y2]` of the chart where the metric appears.
    * **Cross-Verification**: Require the model to verify whether the visual chart matches the accompanying textual footnotes.
    * **Direct Image Display**: Render the retrieved image crop directly in the user chat interface alongside the textual answer.
    """)

st.divider()

# Section 4: Interactive Workbench — Multimodal Enterprise RAG Simulator
st.header("4. Interactive Workbench: Multimodal Document Retrieval & Synthesis Simulator", divider="rainbow")

st.markdown("""
Test Multimodal RAG on a simulated **Quarterly Cloud Enterprise Earnings Report**. 
This page contains an **Executive Prose Summary**, an **Operating Segment Table**, and a **Cloud Infrastructure Architecture Diagram**.
Select a query to see how different modalities are retrieved and synthesized.
""")

sample_multimodal_doc = {
    "doc_id": "SEC_10Q_Q3_2024",
    "title": "Acme Cloud Corp - Q3 2024 Financial & Technical Review",
    "text_chunks": [
        "Executive Summary: In Q3 2024, Acme Cloud achieved record consolidated revenue of $12.4 Billion, driven primarily by accelerated enterprise adoption of Generative AI inference clusters and private VPC networking services.",
        "Risk Factors: Supply chain constraints in high-bandwidth memory (HBM3e) GPU allocations may impact deployment lead times across EMEA availability zones during Q4 2024."
    ],
    "table_data": pd.DataFrame({
        "Business Segment": ["Enterprise Cloud Infrastructure", "AI Model Hosting & API", "Cybersecurity & Identity", "Developer Tooling & DevOps"],
        "Q3 2023 Revenue ($M)": [4200, 850, 1100, 650],
        "Q3 2024 Revenue ($M)": [5800, 2400, 1350, 850],
        "YoY Growth Rate": ["+38.1%", "+182.4%", "+22.7%", "+30.8%"]
    }),
    "visual_diagram": {
        "title": "Global Multi-Region AI Inference Architecture",
        "description": "System architecture diagram illustrating the load-balancer distributing inference requests across US-East, US-West, and EU-Central GPU clusters with Redis semantic caching layers.",
        "key_visual_entities": ["Kubernetes Ingress", "GPU H100 Cluster", "Redis Semantic Cache", "Cross-Region S3 Storage"]
    }
}

multimodal_queries = {
    "Tabular Financial Query": "Which business segment had the highest year-over-year revenue growth in Q3 2024?",
    "Visual Architecture Diagram Query": "How does the global inference architecture cache repeated prompts across regions?",
    "Textual Risk & Infrastructure Query": "What are the primary supply chain risk factors affecting GPU deployments?",
    "Cross-Modal Compound Query": "Summarize total cloud infrastructure revenue and describe how requests flow to GPU clusters"
}

selected_query_label = st.selectbox(
    "Choose a Multimodal Evaluation Query:",
    options=list(multimodal_queries.keys())
)

active_mm_q = multimodal_queries[selected_query_label]

col_doc1, col_doc2 = col_doc1, col_doc2 = st.columns([1, 1])

with col_doc1:
    st.subheader("Simulated Enterprise Document Modalities")
    
    st.markdown("**1. Prose Text Chunks:**")
    for tc in sample_multimodal_doc["text_chunks"]:
        st.caption(f"• {tc}")
        
    st.markdown("**2. Structured Tabular Grid (Q3 Segment Performance):**")
    st.dataframe(sample_multimodal_doc["table_data"], use_container_width=True, hide_index=True)
    
    st.markdown("**3. Visual Diagram Metadata:**")
    st.info(f"**Diagram**: `{sample_multimodal_doc['visual_diagram']['title']}`  \n**Content**: {sample_multimodal_doc['visual_diagram']['description']}")

with col_doc2:
    st.subheader("Multimodal Retrieval & Grounded Synthesis")
    st.markdown(f"**Query**: `{active_mm_q}`")
    
    # Simulated routing and retrieval response
    if "Tabular" in selected_query_label:
        st.success("**Modality Retrieved**: Structured Table Grid `[Table_Segment_Q3]`")
        st.markdown("""
        **Grounded Multimodal Answer**:
        According to the Q3 2024 Segment Performance table:
        * **AI Model Hosting & API** experienced the highest year-over-year growth, surging by **+182.4%** (from $850M in Q3 2023 to $2,400M in Q3 2024).
        * The second fastest growing division was Enterprise Cloud Infrastructure with **+38.1%** YoY growth.
        *(Source: `SEC_10Q_Q3_2024 - Table: Operating Segment Review`)*
        """)
    elif "Visual Architecture" in selected_query_label:
        st.success("**Modality Retrieved**: Visual Architecture Diagram `[Diag_Inference_VPC]`")
        st.markdown("""
        **Grounded Multimodal Answer**:
        Based on the Global Multi-Region AI Inference Architecture diagram:
        * Inbound user requests pass through the regional Kubernetes Ingress.
        * Prior to hitting GPU H100 clusters, requests query a **Redis Semantic Caching Layer**.
        * Identical or semantically equivalent prompts are served directly from cache at sub-10ms latency, bypassing GPU forward passes and reducing cross-region network overhead.
        *(Source: `SEC_10Q_Q3_2024 - Figure 3.1: Global AI Architecture`)*
        """)
    elif "Textual Risk" in selected_query_label:
        st.success("**Modality Retrieved**: Prose Paragraph `[Text_Risk_Factors]`")
        st.markdown("""
        **Grounded Multimodal Answer**:
        The primary supply chain risk cited in the report is **allocation constraints in High-Bandwidth Memory (HBM3e) GPU components**.
        This risk is specifically projected to impact hardware deployment lead times across EMEA (Europe, Middle East, and Africa) availability zones in Q4 2024.
        *(Source: `SEC_10Q_Q3_2024 - Section: Risk Factors`)*
        """)
    else:
        st.success("**Modality Retrieved**: Composite Retrieval (Table + Diagram + Prose)")
        st.markdown("""
        **Grounded Multimodal Answer**:
        1. **Revenue Performance**: Enterprise Cloud Infrastructure generated **$5,800 Million** in Q3 2024 (+38.1% YoY growth).
        2. **GPU Traffic Flow**: Inbound traffic hits the regional Ingress load balancer, checks the Redis Semantic Cache for prior hit embeddings, and routes uncached queries to the GPU H100 cluster across US-East, US-West, or EU-Central.
        *(Source: `SEC_10Q_Q3_2024 - Table 1 & Figure 3.1`)*
        """)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: What is ColPali and why does it represent a paradigm shift in PDF document retrieval?", expanded=True):
    st.markdown("""
    **Answer:**
    Traditionally, retrieving information from complex PDFs required an error-prone multi-step pipeline:
    1. OCR (Optical Character Recognition) to extract text strings.
    2. Heuristic layout parsing to detect tables, reading order, and columns.
    3. Vision models or VLMs to generate textual captions for charts.
    4. Text embedding of the resulting extracted text.
    Every step in this chain compounds errors: bad OCR destroys numbers, multi-column layouts get scrambled, and chart visual details are lost.
    
    **ColPali (Févry et al., 2024)** replaces this entire chain with **Vision-Native Retrieval**:
    * Each PDF page is rendered directly as a raw image screenshot (e.g., 1024x1024 pixels).
    * A Vision-Language foundation model (PaliGemma, built on SigLIP and Gemma) encodes the page image into a grid of **multi-vector patch embeddings** (e.g., ~1024 vectors per page).
    * When a user queries with text tokens, ColPali computes **ColBERT Late Interaction (MaxSim)** between query tokens and the visual patch tokens.
    * The query tokens attend directly to the visual pixels representing chart axes, table cells, or bold headings. Zero OCR is needed, preserving 100% of the visual layout and tabular geometry.
    """)

with st.expander("Q2: Why do standard character chunkers fail on financial tables, and how should tables be indexed?"):
    st.markdown("""
    **Answer:**
    Standard text chunkers (e.g., splitting every 500 characters) are blind to table syntax. 
    When applied to tables, chunk boundaries split halfway across rows or detach column headers from numerical cells, rendering the numbers devoid of meaning.
    
    **Production Best Practices for Table Indexing**:
    1. **Structural Table Extraction**: Use layout models (such as Table Transformer - TATR, Microsoft Azure Document Intelligence, or Unstructured.io) to extract tables as intact semantic blocks.
    2. **HTML Serialization**: Represent the table using semantic HTML `<table>` tags (`<th>`, `<tr>`, `<td>`, `rowspan`), which preserve multi-level hierarchical headers better than flat Markdown.
    3. **Row Linearization**: For dense vector matching, linearize each row into an explicit declarative sentence containing all parent column headers (*"For Segment Cloud, Q3 2024 Revenue was $5.8B"*).
    4. **Dual-Store / Metadata Preservation**: Compute embeddings on the linearized summary or VLM description, while storing the raw pristine HTML table in the vector metadata payload for full injection into the synthesis prompt.
    """)

with st.expander("Q3: What are the latency and cost trade-offs of Pipeline 1 (OCR + VLM Captioning) vs. Pipeline 3 (ColPali)?"):
    st.markdown("""
    **Answer:**
    * **Pipeline 1 (OCR + Captioning)**:
      * *Ingestion*: Extremely slow and expensive. Calling a VLM (e.g., GPT-4o) to describe every chart and table in a 1,000-page document costs significant money and takes hours.
      * *Retrieval*: Fast (<10ms) and low memory, because each chunk is a standard 1536-d text vector.
    * **Pipeline 3 (ColPali)**:
      * *Ingestion*: Fast and inexpensive. Simply renders the PDF page to an image and runs a single forward pass through PaliGemma on GPU.
      * *Retrieval*: Higher memory footprint and query latency. Because ColPali uses multi-vector late interaction (1,024 vectors per page), storing vectors for 100,000 pages requires 100M vectors (~100 GB RAM with binary or 4-bit scalar quantization).
    """)

with st.expander("Q4: How do you evaluate a Multimodal RAG pipeline end-to-end?"):
    st.markdown("""
    **Answer:**
    End-to-end evaluation combines retrieval benchmarks and synthesis grounding metrics:
    1. **Visual Document Retrieval Metrics**:
       * **Recall@k and MRR**: Evaluate on multimodal benchmarks like **DocVQA** and **ViDoRe (Visual Document Retrieval Benchmark)**, measuring whether the correct visual page is retrieved for questions requiring chart or table understanding.
    2. **Faithfulness & Visual Grounding**:
       * Use automated LLM-as-a-judge frameworks (such as Ragas or TruLens adapted for vision) to verify that every numeric claim in the synthesized answer matches the specific visual quadrant or table cell.
    3. **Entity Preservation**: Check that exact figures (currencies, percentages, component IDs) were transcribed from the visual page into the answer without rounding or distortion.
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Downsampling PDF Screenshots Aggressively**: Resizing a dense 8.5x11 PDF page down to 224x224 pixels. At 224x224, 8pt table font becomes illegible blurred pixels, making it impossible for vision encoders or VLMs to read numbers. Always maintain at least 1024x1024 or use dynamic patch tiling.
* **Trap 2: Stripping Whitespace and Column Separators in OCR**: Naive text extraction strips column indentation, joining Column A and Column B into a single continuous sentence (*"Revenue 2023 Revenue 2024 100M 250M"*).
* **Trap 3: Feeding Full Multi-Page PDFs into VLMs for Every User Turn**: Passing the entire 50-page PDF image sequence into GPT-4o for every chat question costs over $0.50 per query and adds 5+ seconds latency. Always use a fast retrieval stage (ColPali or Dual-Store) to filter down to the top 2-3 pages before calling the generative VLM.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: ColPali"):
        st.markdown("**Front**: What makes ColPali unique in PDF search?")
        st.divider()
        st.markdown("**Back**: It treats PDF pages as images, bypassing OCR entirely, and uses ColBERT late interaction between text query tokens and visual patch tokens.")

with card_cols[1]:
    with st.popover("Card 2: Table Chunking"):
        st.markdown("**Front**: Why do standard chunkers fail on tables?")
        st.divider()
        st.markdown("**Back**: Arbitrary character length limits cut across rows, separating numbers from their parent column headers and destroying tabular context.")

with card_cols[2]:
    with st.popover("Card 3: Row Linearization"):
        st.markdown("**Front**: What is Table Row Linearization?")
        st.divider()
        st.markdown("**Back**: Converting each table row into an independent declarative sentence that includes all column headers for accurate dense vector matching.")

with card_cols[3]:
    with st.popover("Card 4: Visual Grounding"):
        st.markdown("**Front**: How does a Vision LLM cite visual sources?")
        st.divider()
        st.markdown("**Back**: By referencing specific image crop IDs, diagram figure numbers, or visual quadrant coordinates corresponding to the evidence.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Multimodal RAG:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the 3 Multimodal RAG architectures (OCR+VLM, Dual-Store, ColPali).", key="chk27_3_1")
    st.checkbox("I know why Vision-Native retrieval (ColPali) eliminates fragile OCR pipelines.", key="chk27_3_2")
    st.checkbox("I can implement table serialization strategies (HTML, Markdown, Row Linearization).", key="chk27_3_3")
with c2:
    st.checkbox("I understand how to construct multimodal payloads with base64 images for GPT-4o.", key="chk27_3_4")
    st.checkbox("I know the latency and memory footprint trade-offs of multi-vector page embeddings.", key="chk27_3_5")
    st.checkbox("I can evaluate multimodal RAG systems using DocVQA and ViDoRe benchmarks.", key="chk27_3_6")
