# Prompt Engineering for Machine Learning — Complete Syllabus / Roadmap

---

## 1. Introduction — Prompt Engineering in the ML Workflow

### 1.1 Where Prompts Fit in ML Pipelines
- LLMs as intelligent ML pipeline components — not just chatbots
- Replacing brittle rule-based logic with LLM prompts
- Accelerating ML projects — data labeling, cleaning, EDA, explanation
- Prompts as the interface between structured ML and LLM capabilities
- Difference from Gen AI use — focus is on tabular data, model quality, not generation

### 1.2 High-Value Use Cases for Prompts in ML
- **Data labeling** — generate training labels at scale
- **Feature extraction** — extract structured features from text
- **Data cleaning** — fix, normalize, and validate data via LLM
- **EDA assistance** — summarize, describe, and flag anomalies
- **Model explanation** — translate SHAP, metrics into natural language
- **Synthetic data** — augment minority class or scarce data

### 1.3 Prompt Design Mindset for ML
- ML requires consistency — `temperature=0` for almost all ML tasks
- Output must be parseable — JSON mode or Pydantic always
- Eval is mandatory — measure label quality, extraction accuracy
- Cost awareness — batch API, caching, right model selection
- Validate every LLM output before using as ML input

---

## 2. Prompts for Data Labeling

### 2.1 Binary Classification Labeling
```
You are a data labeling expert.
Classify the following customer review as 'positive' or 'negative'.
Return ONLY the label. No explanation.

Review: "{text}"
Label:
```

### 2.2 Multi-class Labeling
```
Classify the support ticket into exactly one category:
- billing
- technical
- shipping
- returns
- general

Return ONLY the category name.

Ticket: "{text}"
Category:
```

### 2.3 Few-Shot Labeling Prompt
```
Classify customer intent. Examples:

"I want to cancel my subscription" → cancellation
"My payment failed yesterday" → billing_issue
"The app keeps crashing" → technical_bug
"Where is my order?" → order_status

Now classify: "{text}"
Intent:
```

### 2.4 Structured Label with Metadata
```python
class Label(BaseModel):
    category: Literal['billing', 'technical', 'shipping', 'returns', 'general']
    confidence: Literal['high', 'medium', 'low']
    ambiguous: bool

response = client.beta.chat.completions.parse(
    model='gpt-4o-mini',
    messages=[
        {'role': 'system', 'content': 'Label support tickets accurately.'},
        {'role': 'user', 'content': f'Label this ticket: {text}'}
    ],
    response_format=Label,
    temperature=0
)
```

### 2.5 Handling Ambiguous Labels
- `confidence: 'low'` or `ambiguous: true` → human review queue
- Include `'If the text is ambiguous, set ambiguous=true'` in prompt
- Route low-confidence samples to active learning loop

### 2.6 Batch Labeling Strategy
- Use Batch API for datasets > 1k samples — 50% cost savings
- Format each sample as a JSONL line
- Download results → validate → load into DataFrame

---

## 3. Prompts for Feature Extraction

### 3.1 Single Feature Extraction
```
Extract the delivery date mentioned in the customer message.
Return ONLY the date in YYYY-MM-DD format.
If no date is mentioned, return null.

Message: "{text}"
Date:
```

### 3.2 Multi-feature Extraction
```python
class ProductFeatures(BaseModel):
    name: str | None
    brand: str | None
    price: float | None
    color: str | None
    defect_mentioned: bool
    sentiment: Literal['positive', 'negative', 'neutral']

response = client.beta.chat.completions.parse(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': f'Extract product info: {text}'}],
    response_format=ProductFeatures,
    temperature=0
)
```

### 3.3 Numeric Feature Extraction
- Urgency score 1–5 — `'Rate urgency from 1 (low) to 5 (emergency). Return only the integer.'`
- Sentiment score — `'Rate sentiment from 0 (very negative) to 1 (very positive). Return a float.'`
- Complexity estimate — `'Estimate resolution complexity: 1=simple, 2=moderate, 3=complex.'`

### 3.4 Nested Structure Extraction
```python
class Entity(BaseModel):
    text: str
    type: Literal['person', 'organization', 'location', 'date', 'product']

class Extraction(BaseModel):
    entities: list[Entity]
    main_topic: str
    action_required: bool
```

### 3.5 Validation and Fallback
```python
def extract_features(text):
    try:
        response = client.beta.chat.completions.parse(
            model='gpt-4o-mini',
            messages=[{'role':'user','content': f'Extract: {text}'}],
            response_format=MySchema,
            temperature=0
        )
        return response.choices[0].message.parsed
    except Exception:
        return None  # Trigger human review or fallback rule

df['features'] = df['text'].apply(extract_features)
df_valid = df[df['features'].notna()]
```

---

## 4. Prompts for Data Cleaning

### 4.1 Typo and OCR Error Correction
```
Fix spelling and OCR errors in the following text.
Return ONLY the corrected text. Make no other changes.

Text: "{text}"
Corrected:
```

### 4.2 Standardizing Inconsistent Values
```
Normalize the following company name to a standard format (Title Case, remove legal suffixes like Ltd, Inc, Corp).
Return ONLY the normalized name.

Name: "{name}"
Normalized:
```

### 4.3 Date Normalization
```
Convert the following date expression to YYYY-MM-DD format.
If you cannot determine the date, return null.

Date expression: "{date_str}"
ISO date:
```

### 4.4 Duplicate Detection
```
Are the following two company names referring to the same organization?
Return ONLY 'yes' or 'no'.

Name 1: "{name1}"
Name 2: "{name2}"
Same organization:
```

### 4.5 Missing Value Imputation Prompt
```
Given this customer profile, what is the most likely value for the missing field "{field}"?
Return ONLY the value. If uncertain, return null.

Profile: {json.dumps(profile)}
Missing field: {field}
Value:
```

---

## 5. Prompts for Exploratory Data Analysis (EDA)

### 5.1 Column Description Generation
```python
df_info = {
    'columns': df.dtypes.to_dict(),
    'sample': df.head(5).to_dict(),
    'stats': df.describe().to_dict()
}

prompt = f"""
You are a data analyst. Given this dataset information:
{json.dumps(df_info, default=str)}

Describe:
1. What this dataset likely represents
2. Potential data quality issues in each column
3. Which columns are likely most predictive for a churn model
4. Suggested feature engineering steps

Be specific and concise.
"""
```

### 5.2 Anomaly Description
```
The following values appear to be outliers in the "{column}" column:
{outliers}

Explain why these values might be anomalies and suggest whether to:
a) Remove them
b) Cap them
c) Keep them

Justify your recommendation in 2 sentences.
```

### 5.3 Feature Suggestion
```
I am building a model to predict customer churn.
My current features are: {', '.join(feature_names)}

Suggest 5 additional features I could engineer from existing data that would likely improve model performance.
For each feature, provide:
- Name
- Calculation method
- Why it might be predictive

Return as JSON array.
```

### 5.4 Class Imbalance Explanation
```
My binary classification dataset has:
- Class 0: {n_neg} samples ({neg_pct:.1f}%)
- Class 1: {n_pos} samples ({pos_pct:.1f}%)

Explain the implications of this imbalance and recommend:
1. Metric to use instead of accuracy
2. Sampling strategy
3. Model parameter to set

Be specific with implementation details.
```

---

## 6. Prompts for Model Explanation

### 6.1 SHAP Value Explanation for Non-technical Audience
```
A machine learning model predicted that customer "{customer_id}" will churn.
The top factors influencing this prediction (SHAP values):
{shap_table}

Explain this prediction in 2-3 sentences for a non-technical business stakeholder.
Focus on what they can act on to prevent churn.
Do not mention SHAP or technical ML terms.
```

### 6.2 Feature Importance Narrative
```
A churn prediction model's most important features are:
{feature_importance_table}

Write a brief (3-4 sentence) business narrative explaining what the model learned
about why customers churn. Use business language, not ML jargon.
```

### 6.3 Error Analysis Prompt
```
These customer reviews were incorrectly classified as positive by a sentiment model:
{misclassified_examples}

Analyze these examples and identify:
1. Common patterns that confused the model
2. Linguistic features the model likely missed
3. Suggestions for improving the training data

Be specific and actionable.
```

### 6.4 Metric Report Generation
```
Generate a concise model performance report for a business audience.

Model: {model_name}
Task: Binary classification for customer churn
Metrics on test set:
- Accuracy: {accuracy:.1%}
- Precision: {precision:.1%}
- Recall: {recall:.1%}
- F1 Score: {f1:.3f}
- AUC-ROC: {auc:.3f}

Write 3-4 sentences summarizing performance, emphasizing business impact.
Do not explain what precision or recall mean. Assume the reader knows.
```

---

## 7. Prompts for Synthetic Data Generation

### 7.1 Minority Class Augmentation
```
Generate {n} realistic customer support tickets for the category "billing_dispute".

Requirements:
- Vary the phrasing, tone, and specific issue
- Include amounts, dates, and account-specific details (make them up)
- Mix formal and informal language
- Keep each between 20-100 words

Return as JSON array of strings.
```

### 7.2 Style-Varied Augmentation
```
Rewrite the following customer review in 5 different styles:
1. Very short (under 10 words)
2. Very formal
3. With typos and informal language
4. With sarcasm
5. In bullet points

Original: "{review}"

Return as JSON array of 5 strings.
```

### 7.3 Quality Filtering Synthetic Data
```
Evaluate whether the following synthetic customer review is realistic and usable for ML training.

Review: "{synthetic_review}"
Category: {category}

Is this review:
1. Realistic for the given category?
2. Grammatically plausible?
3. Free from obvious AI artifacts?

Return JSON: {"usable": bool, "issues": str | null}
```

---

## 8. Prompts for Text Classification (Replacing Rule-Based Systems)

### 8.1 Routing Prompt
```
You are a customer support routing system.
Route the incoming message to the correct team.

Teams:
- billing: payment, invoice, subscription, refund
- technical: bugs, errors, crashes, login, app issues
- sales: pricing, upgrade, demo request, new account
- general: other

Return ONLY the team name.

Message: "{message}"
Team:
```

### 8.2 Hierarchical Classification
```
First classify the main category, then the subcategory.

Main categories and subcategories:
- product_issue: defect, wrong_item, damaged, missing
- service_issue: slow_shipping, rude_agent, no_response
- billing_issue: overcharge, refund_delay, wrong_amount

Return JSON: {"main_category": str, "subcategory": str}

Message: "{text}"
```

### 8.3 Multi-label Classification
```
Tag the following article with ALL applicable topics from this list:
{', '.join(topics)}

Return a JSON array of matching topics. If none match, return an empty array.

Article: "{text}"
Topics:
```

---

## 9. Prompt Evaluation for ML Tasks

### 9.1 Labeling Quality Evaluation
```python
def evaluate_label_quality(texts, llm_labels, human_labels):
    from sklearn.metrics import classification_report, cohen_kappa_score
    print(classification_report(human_labels, llm_labels))
    print('Cohen Kappa:', cohen_kappa_score(human_labels, llm_labels))
```

### 9.2 Extraction Quality Evaluation
```python
def evaluate_extraction(field, predictions, ground_truth):
    # Exact match
    exact = sum(p == g for p, g in zip(predictions, ground_truth))
    print(f'{field} exact match: {exact/len(ground_truth):.2%}')
    # Null rate
    null_rate = sum(p is None for p in predictions)
    print(f'{field} null rate: {null_rate/len(predictions):.2%}')
```

### 9.3 Prompt A/B Testing
```python
def ab_test(prompt_a, prompt_b, eval_set, metric_fn):
    scores_a = [metric_fn(call_llm(prompt_a, ex['input']), ex['expected']) for ex in eval_set]
    scores_b = [metric_fn(call_llm(prompt_b, ex['input']), ex['expected']) for ex in eval_set]
    print(f'Prompt A: {np.mean(scores_a):.3f} ± {np.std(scores_a):.3f}')
    print(f'Prompt B: {np.mean(scores_b):.3f} ± {np.std(scores_b):.3f}')
```

### 9.4 Consistency Testing
```python
def consistency_test(prompt, input_text, n=5):
    responses = [call_llm(prompt, input_text) for _ in range(n)]
    unique = len(set(responses))
    print(f'Unique responses: {unique}/{n}')  # 1/5 = perfectly consistent
    return responses
```

---

## 10. Cost-Efficient Prompting for ML Pipelines

### 10.1 Model Selection by Task
- Simple labeling (sentiment, category) → `gpt-4o-mini`
- Complex extraction (nested, multi-field) → `gpt-4o-mini` with few-shot
- Reasoning-heavy (error analysis, feature suggestion) → `gpt-4o`
- Never use `o1` for ML data tasks — overkill and expensive

### 10.2 Batch API for Large Datasets
- > 1k samples → always use Batch API (50% cheaper)
- 24-hour turnaround acceptable for offline ML pipelines
- Format JSONL → upload → poll → download results

### 10.3 Prompt Length Optimization
- Remove verbose instructions — be concise
- Move few-shot examples to system prompt (cached prefix)
- Use `text-embedding-3-small` not `ada-002` for embeddings (cheaper, better)
- Set `max_tokens` to cap output — extraction needs only few tokens

### 10.4 Response Caching
- Cache deterministic outputs — same input → same prompt → same output
- Hash `(model, prompt, input)` as cache key
- Use Redis or disk cache for production pipelines
- Only cache `temperature=0` outputs — deterministic responses

---

## 11. Prompt Patterns Cheatsheet for ML

### 11.1 Classification
- System: `'You are a classification expert. Return ONLY the label from: {labels}.'`
- User: `'Classify: "{text}"'`
- Format: `response_format={'type':'json_object'}` or plain text

### 11.2 Extraction
- System: `'Extract structured fields. Return valid JSON only. Null for missing fields.'`
- User: `'Extract from: "{text}"\nSchema: {schema}'`
- Format: Pydantic Structured Outputs

### 11.3 Cleaning
- System: `'Fix and normalize the input. Return ONLY the corrected version.'`
- User: `'Fix: "{text}"'`
- Format: plain text

### 11.4 Generation / Augmentation
- System: `'Generate realistic training examples for ML. Vary phrasing and content.'`
- User: `'Generate {n} examples for category: {category}. JSON array of strings.'`
- Format: JSON array

### 11.5 Explanation
- System: `'Explain ML model decisions for a non-technical business audience.'`
- User: `'Prediction: {pred}. Feature importance: {features}. Explain in 2 sentences.'`
- Format: plain text

---

## 12. Interview Topics for ML Engineers

### 12.1 Frequently Asked Questions
- How would you use LLM prompts to label a 100k-row dataset cost-effectively?
- What temperature do you use for data labeling and why?
- How do you validate LLM extraction quality against ground truth?
- What is prompt injection and how does it affect ML pipelines?
- How would you build a prompt A/B testing framework?

### 12.2 ML-Oriented Coding Problems
- Write a few-shot prompt for multi-class intent classification
- Build a Pydantic extraction schema for product info from raw text
- Implement a label quality evaluation script (accuracy + kappa)
- Design a cost-efficient batch labeling pipeline using OpenAI Batch API
- Build a prompt A/B test with LLM-as-judge scoring

---

## 13. Real-World Projects

### 13.1 Beginner Projects
- Zero-shot sentiment labeler for tweet dataset
- Product attribute extractor from e-commerce descriptions
- Simple EDA assistant — describe dataset, flag issues

### 13.2 Intermediate Projects
- Full labeling pipeline — Batch API + quality eval + human review routing
- Multi-field extraction pipeline with Pydantic + validation
- SHAP + LLM explanation generator for churn model

### 13.3 Advanced Projects
- Active learning loop — LLM labels easy samples, human labels hard ones
- Prompt A/B testing framework with statistical significance test
- Automated feature suggestion and EDA report generation tool
