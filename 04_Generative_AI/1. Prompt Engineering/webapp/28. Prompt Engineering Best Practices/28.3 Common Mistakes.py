import streamlit as st
import json

st.set_page_config(
    page_title="28.3 Common Mistakes - Prompt Engineering",
    page_icon="⚠️",
    layout="wide"
)

# Header Section
st.title("28.3 Common Mistakes, Anti-Patterns, & How to Avoid Them")
st.caption("Chapter 28: Prompt Engineering Best Practices | Domain: 1. Prompt Engineering")
st.markdown("""
Prompt engineering failures in enterprise systems are rarely unpredictable flukes; they almost always trace back to 
**seven pervasive engineering anti-patterns**: vague directives, instruction dilution, conflicting constraints, 
exemplar starvation, unvalidated outputs, blind trust in completions, and 'vibes-based' development without automated evaluation.
""")

st.divider()

# Section 1: The 7 Deadly Prompt Engineering Anti-Patterns
st.header("1. The Seven Deadly Prompt Anti-Patterns", divider="rainbow")
st.markdown("""
Understanding these failure modes and their concrete remedies is what separates novice prompters from senior AI engineers:
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. The Vague Directive Anti-Pattern")
    st.markdown("""
    * **Mistake**: Using subjective or polysemous instructions (*"Make this better", "Write a good memo"*).
    * **Consequence**: High variance; the model picks arbitrary interpretations across runs.
    * **Remedy**: Quantified metrics (*"Rewrite in under 80 words, 3 bullets, executive tone"*).
    """)

    st.subheader("2. Instruction Dilution (Overload)")
    st.markdown("""
    * **Mistake**: Cramming 25 disparate rules into a single prompt block.
    * **Consequence**: Transformer attention heads dilute focus; the model randomly drops rules in the middle.
    * **Remedy**: Decompose into a 2-step prompt chain or hierarchical XML sections.
    """)

    st.subheader("3. Constraint Collision")
    st.markdown("""
    * **Mistake**: Mutually impossible demands (*"Provide an exhaustive deep-dive explanation in under 40 words"*).
    * **Consequence**: Unpredictable compromises and hallucinations.
    * **Remedy**: Establish explicit priority rankings between competing constraints.
    """)

    st.subheader("4. Exemplar Starvation")
    st.markdown("""
    * **Mistake**: Explaining a complex, nuanced formatting syntax entirely through verbose prose without examples.
    * **Consequence**: Formatting drift and downstream parser crashes.
    * **Remedy**: Anchor with 2 to 3 ground-truth few-shot demonstrations.
    """)

with col2:
    st.subheader("5. Ignoring Programmatic Output Validation")
    st.markdown("""
    * **Mistake**: Passing raw LLM completion strings directly to SQL databases, users, or downstream microservices.
    * **Consequence**: Uncaught JSONDecodeErrors, SQL injection risks, and service crashes.
    * **Remedy**: Pydantic schema validation wrappers with auto-healing retry loops.
    """)

    st.subheader("6. Blind Trust in Factual Completions")
    st.markdown("""
    * **Mistake**: Assuming an LLM will never hallucinate citations, URLs, or arithmetic.
    * **Consequence**: Critical compliance, legal, and reputational disasters.
    * **Remedy**: Grounding in RAG contexts + automated verification checks (CoVe).
    """)

    st.subheader("7. Vibes-Based Engineering (No Evals)")
    st.markdown("""
    * **Mistake**: Testing a prompt on 3 ad-hoc queries in the playground and shipping to production.
    * **Consequence**: Silent regressions when rolling out updates.
    * **Remedy**: Golden benchmark datasets with automated CI/CD evaluation scoring.
    """)

st.divider()

# Section 2: Anti-Pattern Autopsy Table
st.header("2. Anti-Pattern Autopsy & Remediation Matrix", divider="rainbow")
st.markdown("Detailed breakdown of symptoms, root causes, and production remedies:")

st.table([
    {
        "Anti-Pattern": "Vague Directives",
        "Novice Symptom": "Model output varies wildy between runs",
        "Engineering Fix": "Quantify constraints, tone, and format bounds"
    },
    {
        "Anti-Pattern": "Instruction Overload",
        "Novice Symptom": "Model obeys rule 1 and 2, but ignores rule 8",
        "Engineering Fix": "Decompose into sequential prompt chain"
    },
    {
        "Anti-Pattern": "Constraint Collision",
        "Novice Symptom": "Model outputs truncated or awkward half-sentences",
        "Engineering Fix": "Remove contradictions; prioritize rules"
    },
    {
        "Anti-Pattern": "Exemplar Starvation",
        "Novice Symptom": "Downstream JSON parsers crash intermittently",
        "Engineering Fix": "Provide 2 concrete few-shot input/output examples"
    },
    {
        "Anti-Pattern": "Unvalidated Output",
        "Novice Symptom": "500 Server Errors in backend services",
        "Engineering Fix": "Enforce strict Pydantic parsing with auto-retry"
    },
    {
        "Anti-Pattern": "Vibes-Based Dev",
        "Novice Symptom": "Prompt update fixes bug A but breaks cases B and C",
        "Engineering Fix": "Automated regression testing against 100+ golden cases"
    }
])

st.divider()

# Section 3: Interactive Prompt Anti-Pattern Scanner
st.header("3. Interactive Prompt Anti-Pattern Scanner", divider="rainbow")
st.markdown("Paste any prompt to run a static anti-pattern diagnostic:")

test_input = st.text_area(
    "Paste Candidate Prompt:",
    value="""You are a very helpful assistant. Please review this document and provide an exhaustive, in-depth, completely thorough analysis of every single risk factor. 
Also keep your response extremely brief, ideally under 30 words. 
Output as pure JSON, but be sure to explain your reasoning thoroughly before the JSON block.""",
    height=120
)

if st.button("Scan for Anti-Patterns", type="primary"):
    flags = []
    
    if "exhaustive" in test_input.lower() and ("brief" in test_input.lower() or "under 30 words" in test_input.lower()):
        flags.append(("Constraint Collision", "Demands both 'exhaustive in-depth analysis' and 'under 30 words'. The model will fail one constraint."))
        
    if "pure json" in test_input.lower() and "explain your reasoning" in test_input.lower():
        flags.append(("Format Conflict", "Demands 'pure JSON' while requesting pre-explanation prose. High probability of JSON parsing errors."))
        
    if "very helpful assistant" in test_input.lower() or "please review" in test_input.lower():
        flags.append(("Vague Directives", "Contains conversational filler ('very helpful assistant') lacking precise persona or domain rules."))
        
    if len(flags) > 0:
        st.warning(f"Detected {len(flags)} Critical Anti-Patterns:")
        for title, desc in flags:
            st.error(f"**{title}**: {desc}")
    else:
        st.success("🎉 No common anti-patterns detected in this static scan!")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa1, qa2 = st.columns(2)

with qa1:
    with st.expander("Q: What is 'Vibes-Based Engineering' and why does it destroy production AI applications?"):
        st.write("""
        'Vibes-based engineering' refers to evaluating prompt quality through subjective manual spot-checking 
        (e.g., a developer trying 2 queries in a UI and declaring 'looks great!'). 
        LLM behavior across high-dimensional token spaces is non-linear: fixing a prompt for one test case 
        frequently degrades performance across 10 unseen edge cases. 
        Without automated golden dataset evals in CI/CD, teams enter an endless cycle of manual prompt tweaking 
        and silent production regressions.
        """)

    with st.expander("Q: How do you identify Instruction Dilution and what is the refactoring strategy?"):
        st.write("""
        Instruction dilution is diagnosed when an LLM consistently obeys instructions placed at the beginning 
        and end of a prompt, but selectively ignores constraints in the middle. 
        The refactoring strategy is:
        1. Prune redundant wording.
        2. Group remaining rules into numbered, XML-delimited categories.
        3. If instruction count exceeds ~10 distinct rules, split the task into a sequential 2-stage prompt pipeline.
        """)

with qa2:
    with st.expander("Trap to Avoid: Trusting Self-Reported Model Confidence"):
        st.write("""
        Asking an LLM *"Rate your confidence from 1 to 10"* does not produce calibrated probabilities. 
        Models frequently assign '10/10 confidence' to total hallucinations because their training rewards 
        authoritative tone. Use log probabilities (logprobs), ensemble consensus (Self-Consistency), 
        or external verification checkers rather than self-reported confidence claims.
        """)

    with st.expander("Flashcards: Anti-Pattern Terms"):
        fc_sel = st.selectbox(
            "Select Anti-Pattern Term:",
            ["Vibes-Based Dev", "Constraint Collision", "Instruction Dilution", "Exemplar Starvation"]
        )
        if fc_sel == "Vibes-Based Dev":
            st.info("Relying on ad-hoc human impressions rather than automated metric evaluations on golden sets.")
        elif fc_sel == "Constraint Collision":
            st.info("Including contradictory requirements that force the model to violate constraints.")
        elif fc_sel == "Instruction Dilution":
            st.info("Overwhelming attention heads with too many rules, causing intermediate rules to be ignored.")
        else:
            st.info("Attempting complex formatting tasks without few-shot input/output examples.")

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I can recognize and resolve constraint collisions in prompt instructions.")
    st.checkbox("I know why prompt updates must be evaluated against automated golden test sets.")
with c2:
    st.checkbox("I never expose unvalidated LLM output strings directly to downstream services.")
    st.checkbox("I understand why asking models for self-reported confidence is uncalibrated.")
