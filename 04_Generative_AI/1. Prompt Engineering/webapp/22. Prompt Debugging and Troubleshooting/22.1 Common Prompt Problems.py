import streamlit as st
import json

st.set_page_config(
    page_title="22.1 Common Prompt Problems - Prompt Engineering",
    page_icon="🔍",
    layout="wide"
)

# Header Section
st.title("22.1 Common Prompt Problems & Pathologies")
st.caption("Chapter 22: Prompt Debugging and Troubleshooting | Domain: 1. Prompt Engineering")
st.markdown("""
When prompt-driven systems fail in production, developers often assume the model has failed. 
In practice, over 85% of failures stem from **structural prompt pathologies**: ambiguous specifications, 
context starvation, mutually contradictory constraints, unparsed schema drift, hallucinations, and high output variance. 
Diagnosing the root cause is the first step toward building deterministic, production-grade applications.
""")

st.divider()

# Section 1: The 6 Root Prompt Pathologies
st.header("1. Taxonomy of Common Prompt Failures", divider="rainbow")
st.markdown("""
Every prompt failure falls into one of six core failure modes. Understanding their signatures allows engineers to isolate and fix bugs rapidly.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("1. Ambiguous Instructions")
    st.markdown("""
    * **Symptom**: The model gives vastly different responses across runs or picks unexpected interpretations.
    * **Root Cause**: Polysemous words, missing audience definitions, or unspecified scope.
    * **Example**: `"Make this text better"` vs `"Rewrite for a technical director, reducing passive voice by 50%"`.
    """)

with col2:
    st.subheader("2. Missing Context")
    st.markdown("""
    * **Symptom**: Model invents missing domain rules or assumes generic defaults.
    * **Root Cause**: Assuming the LLM possesses private knowledge, session state, or organization definitions.
    * **Example**: Asking for refund calculations without providing company refund tier charts.
    """)

with col3:
    st.subheader("3. Conflicting Instructions")
    st.markdown("""
    * **Symptom**: The model randomly obeys one rule and ignores another.
    * **Root Cause**: Over-constraining (e.g. `"Provide an exhaustive analysis in under 50 words"`).
    * **Example**: Asking for raw JSON output while simultaneously requesting markdown explanations.
    """)

col4, col5, col6 = st.columns(3)
with col4:
    st.subheader("4. Formatting Drift & Parse Errors")
    st.markdown("""
    * **Symptom**: Downstream services crash on `json.loads()`.
    * **Root Cause**: Model injects conversational filler (e.g., `"Sure! Here is the JSON:"`), markdown code fences, or trailing commas.
    * **Fix**: Pydantic schema enforcement or grammar-constrained decoding.
    """)

with col5:
    st.subheader("5. Factual Hallucinations")
    st.markdown("""
    * **Symptom**: Fabricated citations, nonexistent API endpoints, or phantom statistics stated with high confidence.
    * **Root Cause**: Forcing the model to guess when context is empty or penalizing refusal too heavily.
    * **Fix**: Explicit negative constraints (`"If context is insufficient, respond 'UNKNOWN'"`).
    """)

with col6:
    st.subheader("6. Inconsistent & Non-Deterministic Outputs")
    st.markdown("""
    * **Symptom**: The application works during testing but breaks in 1 out of 10 customer requests.
    * **Root Cause**: High temperature, non-zero top-p, or loose syntactic structure without anchor delimiters.
    * **Fix**: Set `temperature=0.0` and utilize rigid few-shot exemplars.
    """)

st.divider()

# Section 2: Before-and-After Autopsy of Buggy Prompts
st.header("2. Autopsy of Production Prompt Bugs", divider="rainbow")
st.markdown("Examine real production prompts, why they broke, and the hardened corrections.")

bug_type = st.selectbox(
    "Select Failure Case to Inspect:",
    [
        "Bug 1: Conflicting Instructions (JSON vs Conversational)",
        "Bug 2: Missing Domain Context (The Phantom Tier Bug)",
        "Bug 3: Ambiguous Extraction (Entity Boundary Drift)"
    ]
)

if bug_type == "Bug 1: Conflicting Instructions (JSON vs Conversational)":
    st.markdown("#### Scenario: API Endpoint Needs Raw JSON, but Prompt Contains Conflicting Politeness Directives")
    c_bad, c_good = st.columns(2)
    with c_bad:
        st.error("❌ Broken Buggy Prompt")
        st.code('''System: You are an ultra-polite customer service agent. Always greet the user warmly and explain your reasoning clearly before giving your answer.

User: Extract the order ID from this ticket: "My package 98212 was damaged". Output only valid JSON with key "order_id". Do not include any extra text.''', language="text")
        st.caption("Result: Model outputs 'Hello! I would be glad to help. Based on your ticket, here is the JSON: {\"order_id\": \"98212\"}'. Downstream JSON parser throws JSONDecodeError.")
        
    with c_good:
        st.success("✅ Hardened Debugged Prompt")
        st.code('''System: You are a strict JSON extraction microservice. 
You must output ONLY a valid JSON object matching the requested schema. 
Never include greetings, markdown formatting, or preamble.

User:
<ticket>My package 98212 was damaged</ticket>

<json_schema>
{"order_id": "string"}
</json_schema>''', language="text")
        st.caption("Result: Pure, unpolluted `{\"order_id\": \"98212\"}`.")

elif bug_type == "Bug 2: Missing Domain Context (The Phantom Tier Bug)":
    st.markdown("#### Scenario: LLM Estimates Enterprise SLA Without Reference Policy")
    c_bad, c_good = st.columns(2)
    with c_bad:
        st.error("❌ Broken Buggy Prompt")
        st.code('''User: A Gold customer has had an outage for 3 hours. Calculate their SLA credit penalty according to standard SLA guidelines.''', language="text")
        st.caption("Result: Model fabricates a 15% discount based on general ITIL public data, contradicting the company's real 5% policy.")
        
    with c_good:
        st.success("✅ Hardened Debugged Prompt")
        st.code('''System: Calculate SLA penalty credits strictly based on the provided company tier schedule. If the tier or outage time is not covered in the policy, respond with "ESCALATE_TO_BILLING".

<policy>
Gold Tier: 1 to 2 hours outage = 0% credit. 2 to 4 hours outage = 5% credit. > 4 hours = 10% credit.
</policy>

<inquiry>
Customer Tier: Gold
Outage Duration: 3 hours
</inquiry>''', language="text")
        st.caption("Result: Deterministic 5% credit with direct quote from the policy.")

else:
    st.markdown("#### Scenario: Entity Extraction with Ambiguous Boundaries")
    c_bad, c_good = st.columns(2)
    with c_bad:
        st.error("❌ Broken Buggy Prompt")
        st.code('''User: Find the company and the person: "John from Apple Pie Bakery called regarding Tim Cook's presentation."''', language="text")
        st.caption("Result: Model tags 'Apple' as the company and 'Pie Bakery' as a note, splitting compound entities.")
        
    with c_good:
        st.success("✅ Hardened Debugged Prompt")
        st.code('''System: Extract entities into predefined categories using full grammatical boundaries.
Exemplar:
Input: "Sarah at General Mills bought Acme Inc."
Output: {"people": ["Sarah"], "organizations": ["General Mills", "Acme Inc."]}

Input: "John from Apple Pie Bakery called regarding Tim Cook\'s presentation."
Output:''', language="text")
        st.caption("Result: Accurate identification of 'Apple Pie Bakery' and distinct attribution of 'John' vs 'Tim Cook'.")

st.divider()

# Section 3: Diagnostic Root-Cause Flowchart
st.header("3. Diagnostic Root-Cause Troubleshooting Flowchart", divider="rainbow")
st.markdown("Follow this systematic troubleshooting tree whenever an LLM response breaks.")

st.graphviz_chart('''
digraph DiagnosticTree {
    rankdir=TB;
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10];
    
    Start [label="Prompt Failure Detected", fillcolor="#FFCDD2"];
    Q1 [label="Is the error a syntax/JSON crash?", shape=diamond, fillcolor="#FFF9C4"];
    FixFormat [label="Enforce Pydantic schema or\\ngrammar constrained decoding", fillcolor="#C8E6C9"];
    
    Q2 [label="Is the output factually wrong?", shape=diamond, fillcolor="#FFF9C4"];
    Q3 [label="Was true fact provided in context?", shape=diamond, fillcolor="#FFF9C4"];
    FixRAG [label="Fix Retrieval/RAG context injection", fillcolor="#C8E6C9"];
    FixNegative [label="Add strict refusal constraint\\n('Say UNKNOWN if absent')", fillcolor="#C8E6C9"];
    
    Q4 [label="Is failure intermittent (1 in 10)?", shape=diamond, fillcolor="#FFF9C4"];
    FixTemp [label="Lower Temperature to 0.0\\n& Add 2 Few-Shot Exemplars", fillcolor="#C8E6C9"];
    FixSplit [label="Decompose into Prompt Chain\\nor sequential steps", fillcolor="#C8E6C9"];
    
    Start -> Q1;
    Q1 -> FixFormat [label="Yes"];
    Q1 -> Q2 [label="No"];
    
    Q2 -> Q3 [label="Yes"];
    Q3 -> FixRAG [label="No"];
    Q3 -> FixNegative [label="Yes"];
    Q2 -> Q4 [label="No"];
    
    Q4 -> FixTemp [label="Yes"];
    Q4 -> FixSplit [label="No (Task too complex)"];
}
''', use_container_width=False)

st.divider()

# Section 4: Interactive Prompt Linter Workbench
st.header("4. Interactive Prompt Linter & Pathology Scanner", divider="rainbow")
st.markdown("Paste a candidate prompt below to scan for common structural anti-patterns.")

candidate_prompt = st.text_area(
    "Enter Candidate Prompt to Audit:",
    value="""You are a helpful customer assistant. Please analyze this text and make it better. 
Make sure you write an exhaustive report with detailed analysis for every single paragraph.
Also keep the total response under 40 words.
Output only valid JSON, but make sure to explain your reasoning thoroughly first.""",
    height=150
)

if st.button("Run Static Prompt Pathology Audit", type="primary"):
    issues_found = []
    
    # Static heuristic rules
    if "make it better" in candidate_prompt.lower() or "helpful assistant" in candidate_prompt.lower():
        issues_found.append(("Ambiguity Warning", "Vague directives ('make it better', 'helpful assistant') lack concrete success criteria."))
        
    if "exhaustive" in candidate_prompt.lower() and ("under 40 words" in candidate_prompt.lower() or "brief" in candidate_prompt.lower()):
        issues_found.append(("Contradiction Detected", "Prompt requests both 'exhaustive report' and 'under 40 words'. Model will randomly violate one constraint."))
        
    if "only valid json" in candidate_prompt.lower() and "explain your reasoning" in candidate_prompt.lower():
        issues_found.append(("Format Conflict", "Prompt asks for 'only valid JSON' while also requesting conversational explanations. High probability of JSON parse errors."))
        
    if "{" in candidate_prompt and "schema" not in candidate_prompt.lower():
        issues_found.append(("Unanchored JSON", "Curly braces detected without explicit JSON schema definition."))

    if not issues_found:
        st.success("🎉 No obvious structural anti-patterns detected in this static scan!")
    else:
        st.warning(f"Identified {len(issues_found)} Structural Anti-Patterns:")
        for title, desc in issues_found:
            st.error(f"**{title}**: {desc}")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

q_col1, q_col2 = st.columns(2)

with q_col1:
    with st.expander("Q: What causes 'Format Drift' in long LLM interactions and how is it mitigated?"):
        st.write("""
        Format drift occurs when an LLM follows formatting instructions early in a response or session, 
        but gradually lapses into conversational defaults as token generation proceeds or context expands. 
        Mitigation strategies include:
        1. Grammar-constrained generation (e.g. Outlines, vLLM JSON mode)
        2. Schema enforcement via API tool calls
        3. Placing output formatting instructions at the very end of the prompt (recency bias)
        """)

    with st.expander("Q: What is the primary cause of prompt non-determinism despite temperature=0?"):
        st.write("""
        Even with `temperature=0.0`, floating-point non-associativity across distributed GPU tensor parallel operations 
        (e.g., non-deterministic reductions in multi-GPU inference engines like vLLM or TensorRT-LLM) can cause 
        marginal logit differences. If top-ranked tokens have near-identical logits, tie-breaking flips can alter generation. 
        Additionally, dynamic system prompt changes or model backend quantization updates introduce variance.
        """)

with q_col2:
    with st.expander("Trap to Avoid: Blaming Model Size for Context Starvation"):
        st.write("""
        Teams frequently jump from an 8B model to a 70B or 405B model when outputs are poor, assuming the smaller model 
        'isn't smart enough'. In reality, if the ground truth domain rules are missing from the prompt context, 
        even the largest frontier model will hallucinate. Always fix information retrieval and context clarity before upgrading models.
        """)

    with st.expander("Flashcards: Prompt Debugging Terminology"):
        sel_card = st.selectbox(
            "Select Diagnostic Term:",
            ["Context Starvation", "Constraint Collision", "Recency Bias", "Jailbreak Leakage"]
        )
        if sel_card == "Context Starvation":
            st.info("The prompt demands domain-specific logic without injecting the requisite source reference data.")
        elif sel_card == "Constraint Collision":
            st.info("Two mutually impossible requirements in the same prompt (e.g., 'detailed explanation' vs '10 words max').")
        elif sel_card == "Recency Bias":
            st.info("The tendency of LLMs to weigh instructions placed at the very beginning and very end of the prompt more heavily than the middle.")
        else:
            st.info("When untrusted user input contains conflicting instructions that override system prompt constraints.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I can identify the 6 primary categories of prompt failures.")
    st.checkbox("I know how to resolve constraint collisions between JSON schemas and conversational preambles.")
with c2:
    st.checkbox("I understand why temperature=0 does not guarantee 100% bitwise determinism on multi-GPU setups.")
    st.checkbox("I know how to use the diagnostic tree to isolate RAG failures vs hallucination.")
