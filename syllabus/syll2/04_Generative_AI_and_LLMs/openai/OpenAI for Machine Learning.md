# OpenAI API for Machine Learning — Complete Syllabus / Roadmap

---

## 1. Introduction — OpenAI API in ML Workflows

### 1.1 The Role of OpenAI API in Classical ML
- LLMs as intelligent assistants inside ML pipelines
- Automating data annotation and labeling at scale
- Enriching datasets with LLM-generated features
- Generating synthetic training data
- Explaining model predictions in natural language
- Extracting structured features from unstructured text
- Replacing rule-based preprocessing with LLM-based extraction

### 1.2 Where OpenAI Fits in the ML Stack
- **Before training** — data cleaning, labeling, feature extraction, synthetic data
- **During training** — no LLM involvement (handled by sklearn/XGBoost)
- **After training** — explaining predictions, generating reports, building ML-powered apps
- **In production** — LLM as an additional feature source or intelligent post-processor

### 1.3 Installing and Setup
- `pip install openai python-dotenv pandas numpy`
- `.env` file — `OPENAI_API_KEY=sk-...`
- `from dotenv import load_dotenv; load_dotenv()`
- `from openai import OpenAI; client = OpenAI()`

---

## 2. Data Labeling and Annotation with LLMs

### 2.1 Zero-Shot Labeling
```python
def label_sentiment(text):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': 'Label the sentiment: positive, negative, or neutral. Return only one word.'},
            {'role': 'user', 'content': text}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower()

df['label'] = df['review'].apply(label_sentiment)
```

### 2.2 Few-Shot Labeling
- Provide labeled examples in the system or user message
- `f"Examples:\n'Great product!' → positive\n'Terrible quality.' → negative\nNow label: {text}"`
- Improves accuracy on domain-specific categories

### 2.3 Structured Label Output
```python
from pydantic import BaseModel
from typing import Literal

class Label(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral']
    confidence: Literal['high', 'medium', 'low']
    reason: str

response = client.beta.chat.completions.parse(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': f'Label this review: {text}'}],
    response_format=Label
)
label = response.choices[0].message.parsed
```

### 2.4 Batch Labeling with Batch API
- Prepare JSONL with all labeling tasks
- Submit to Batch API — 50% cheaper, 24h turnaround
- Download and parse results
- Ideal for offline dataset labeling at scale

### 2.5 Active Learning Support
- Use LLM to label easy examples confidently
- Flag uncertain examples for human review
- `confidence: 'low'` → add to human annotation queue

---

## 3. Feature Extraction from Unstructured Text

### 3.1 Extracting Structured Fields
```python
class ProductInfo(BaseModel):
    name: str
    price: float | None
    brand: str | None
    category: str

response = client.beta.chat.completions.parse(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': f'Extract product info from: {raw_text}'}],
    response_format=ProductInfo
)
info = response.choices[0].message.parsed
```

### 3.2 Named Entity Extraction
- Extract person names, organizations, locations, dates
- Return as structured JSON
- `'Extract all entities as JSON: {entities: [{text, type}]}'`

### 3.3 Extracting Numeric Features from Text
- Sentiment scores — `float` 0–1
- Urgency level — `int` 1–5
- Topic category — `str` from enum
- Customer intent — `Literal['purchase','refund','inquiry','complaint']`

### 3.4 Multi-field Extraction Pipeline
```python
def extract_features(text):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': 'Extract: sentiment (pos/neg/neu), urgency (1-5), topic (billing/tech/general). JSON only.'},
            {'role': 'user', 'content': text}
        ],
        response_format={'type': 'json_object'},
        temperature=0
    )
    return json.loads(response.choices[0].message.content)

feature_df = df['text'].apply(extract_features).apply(pd.Series)
```

---

## 4. Synthetic Data Generation

### 4.1 Generating Training Examples
```python
def generate_examples(category, n=10):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{
            'role': 'user',
            'content': f'Generate {n} realistic customer reviews for category: {category}. JSON array of strings.'
        }],
        response_format={'type': 'json_object'},
        temperature=0.9
    )
    return json.loads(response.choices[0].message.content)['reviews']
```

### 4.2 Balancing Imbalanced Datasets
- Generate synthetic minority class examples with GPT-4o-mini
- Instruct model to vary phrasing, structure, and style
- Filter generated examples for quality before adding to dataset
- Use `temperature=0.8–1.0` for diversity

### 4.3 Data Augmentation for NLP
- Paraphrase existing training examples
- Back-translation style augmentation via LLM
- Style transfer — formal ↔ informal, long ↔ short
- Add noise — typos, abbreviations, slang

### 4.4 Generating Synthetic Tabular Data
- Describe schema and distribution to GPT-4o
- `'Generate 20 realistic customer records with: age (18-70), income (20k-200k), churn (bool)'`
- Output as JSON → convert to DataFrame
- Use for testing pipelines and demos

---

## 5. Embeddings for ML

### 5.1 Text Embeddings as ML Features
```python
def get_embedding(text):
    response = client.embeddings.create(
        model='text-embedding-3-small',
        input=text
    )
    return response.data[0].embedding

df['embedding'] = df['text'].apply(get_embedding)
X_embed = np.stack(df['embedding'].values)
```

### 5.2 Using Embeddings as Input to sklearn Models
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_embed, y, stratify=y)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

### 5.3 Embeddings for Semantic Search
```python
def find_similar(query, texts, embeddings, top_k=5):
    query_emb = np.array(get_embedding(query))
    sims = cosine_similarity([query_emb], embeddings)[0]
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [(texts[i], sims[i]) for i in top_indices]
```

### 5.4 Dimensionality Reduction on Embeddings
- `PCA(n_components=50).fit_transform(embeddings)` — reduce 1536 → 50
- `TSNE(n_components=2).fit_transform(pca_embeddings)` — visualize in 2D
- `UMAP(n_components=2).fit_transform(embeddings)` — faster than t-SNE
- Color by class label to inspect cluster quality

### 5.5 Embedding Clustering
- `KMeans(n_clusters=10).fit(embeddings)` — cluster documents
- `silhouette_score(embeddings, labels)` — cluster quality
- `DBSCAN(eps=0.3, min_samples=5).fit(embeddings)` — density clusters
- Topics emerge without labeled data

### 5.6 Batch Embeddings for Large Datasets
- Max 2048 inputs per request (batch limit)
- `chunks = [texts[i:i+2048] for i in range(0, len(texts), 2048)]`
- Process chunks and concatenate
- Use Batch API for >100k texts (50% cost reduction)

---

## 6. OpenAI as a Preprocessing Tool for ML

### 6.1 Text Normalization
- Fixing OCR errors — `'Fix spelling in this scanned text: {text}'`
- Normalizing date formats — extract and standardize
- Translating multilingual text to English before ML
- Expanding abbreviations and acronyms

### 6.2 Text Summarization for Feature Engineering
- Summarize long customer reviews to fixed-length representation
- `max_tokens=100` — force concise summary
- Use summarized text as cleaner input for embeddings

### 6.3 Topic and Category Extraction
- `'Assign one of these categories to this text: {categories}. Return only the category name.'`
- Used as a categorical feature in ML models

### 6.4 Sentiment and Intent Scoring
- Sentiment — `float` 0–1 (negative to positive)
- Intent — `Literal['purchase','inquiry','complaint','support']`
- Urgency — `int` 1–5

---

## 7. Model Explanation with LLMs

### 7.1 Explaining SHAP Values in Natural Language
```python
def explain_prediction(shap_values, feature_names, prediction):
    top_features = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)[:5]
    feature_text = '\n'.join([f'{f}: {v:.3f}' for f, v in top_features])
    
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{
            'role': 'user',
            'content': f'The model predicted: {prediction}\nTop SHAP values:\n{feature_text}\nExplain this prediction in 2 sentences for a non-technical audience.'
        }],
        temperature=0.3
    )
    return response.choices[0].message.content
```

### 7.2 Generating Automated ML Reports
- `'Write a brief analysis of these model metrics: Accuracy={acc}, F1={f1}, AUC={auc}'`
- Comparing two model configurations in natural language
- Summarizing cross-validation results for stakeholder report

### 7.3 Error Analysis with LLM
- Feed misclassified examples to GPT-4o
- Ask why they might be difficult to classify
- Identify patterns in errors — domain knowledge extraction

---

## 8. Cost-Efficient Patterns for ML Pipelines

### 8.1 Model Selection by Task Complexity
- `gpt-4o-mini` — labeling, extraction, classification (most ML tasks)
- `gpt-4o` — complex reasoning, multi-step extraction
- `o1` — rarely needed for ML preprocessing tasks
- Always start with `gpt-4o-mini`

### 8.2 Prompt Caching
- Repeated system prompts are cached automatically (same prefix)
- Place static instructions in system message — not user message
- Cache discount applies to repeated identical prefixes

### 8.3 Batch API for ML Data Pipelines
- Labeling 100k+ examples — always use Batch API
- Feature extraction on large corpora
- Embedding generation for large text datasets
- 50% cost reduction, fire-and-forget workflow

### 8.4 Caching Responses Locally
```python
import hashlib, json, os

def cached_completion(prompt, model='gpt-4o-mini'):
    key = hashlib.md5(f'{model}:{prompt}'.encode()).hexdigest()
    cache_file = f'.cache/{key}.json'
    if os.path.exists(cache_file):
        return json.load(open(cache_file))['response']
    response = client.chat.completions.create(
        model=model, messages=[{'role':'user','content':prompt}]
    ).choices[0].message.content
    json.dump({'response': response}, open(cache_file, 'w'))
    return response
```

---

## 9. Building ML-Powered Applications

### 9.1 Text Classification App
- User submits text → LLM extracts features → sklearn model classifies → LLM explains result
- Full pipeline combining classical ML and LLM

### 9.2 Automated EDA with LLM
- Feed DataFrame info to GPT-4o — `df.describe()`, `df.dtypes`, sample rows
- `'Identify potential data quality issues and suggest feature engineering steps'`
- LLM as a first-pass EDA analyst

### 9.3 Data Cleaning Suggestions
- `'This column has values: {unique_values[:50]}. Identify any errors, duplicates, or normalization issues.'`
- Programmatically apply suggestions

### 9.4 Feature Suggestion
- `'Given these features and the task of predicting churn, suggest 5 new features that could improve the model.'`
- Domain-aware feature engineering acceleration

---

## 10. Evaluation of LLM-Assisted ML

### 10.1 Labeling Quality Assessment
- Sample 200 LLM-labeled examples → human review
- Compute agreement — Cohen's Kappa, accuracy vs human labels
- Identify failure categories — edge cases, ambiguous examples

### 10.2 Extraction Quality
- Precision/recall of extracted fields vs ground truth
- `sklearn.metrics.classification_report` per field
- Field-level accuracy

### 10.3 Consistency Testing
- Run same prompt N times — check for variation
- `temperature=0` for deterministic outputs
- Multiple responses and majority vote — `n=5`

### 10.4 Cost Tracking
- Log `response.usage.total_tokens` per request
- Aggregate cost per pipeline step
- Compare cost vs quality at different model tiers

---

## 11. OpenAI API Best Practices for ML

### 11.1 Prompt Design for ML Tasks
- Be specific — define the output format exactly
- Include examples — few-shot for edge cases
- Set `temperature=0` — deterministic labeling
- Use `response_format` — JSON or Pydantic models
- Add `'Return ONLY the answer, no explanation'` — reduce noise

### 11.2 Robustness
- Validate LLM output before using as ML features
- Handle `None` / unexpected values with fallback
- Retry on failure — `try/except openai.RateLimitError`
- Log failed requests and reprocess

### 11.3 Common Mistakes
- Using GPT-4o for simple classification — use mini
- Not setting `temperature=0` for labeling — inconsistent output
- Not validating JSON output before parsing
- Paying for real-time API when Batch API would work
- Including PII in prompts — use anonymization before sending

---

## 12. Interview Topics for ML Engineers

### 12.1 Frequently Asked Questions
- How would you use LLMs to label a dataset at scale?
- What are OpenAI embeddings and how are they used in ML?
- How do you reduce OpenAI API costs in an ML pipeline?
- What is Structured Output and how does it differ from JSON mode?
- How would you evaluate the quality of LLM-generated labels?

### 12.2 ML-Oriented Coding Problems
- Build a sentiment labeling pipeline using GPT-4o-mini with batch processing
- Extract structured product features from raw descriptions using Pydantic
- Use OpenAI embeddings as input to a sklearn classifier and evaluate
- Implement a caching layer for repeated OpenAI API calls in a pipeline
- Generate synthetic training data for a minority class

---

## 13. Real-World ML Projects with OpenAI API

### 13.1 Beginner Projects
- Automated sentiment labeler for tweet dataset
- Product category extractor from raw descriptions
- Text length and sentiment features → sklearn classifier

### 13.2 Intermediate Projects
- Full data labeling pipeline with Batch API + quality evaluation
- Embedding-based semantic search over product catalog
- SHAP + GPT-4o explanation generator for customer churn model

### 13.3 Advanced Projects
- Active learning loop — LLM labels easy examples, human labels uncertain
- End-to-end text ML pipeline — extraction → embedding → sklearn → LLM explanation
- Automated EDA and feature suggestion tool

---

## 14. Roadmap

### 14.1 Beginner Level
- Chat completions for text classification and labeling
- `temperature=0` for consistent outputs
- JSON mode for structured extraction
- Basic embedding generation

### 14.2 Intermediate Level
- Pydantic Structured Outputs for type-safe extraction
- Batch embedding generation for ML features
- Batch API for cost-efficient large-scale labeling
- LLM-based model explanation generation

### 14.3 Advanced Level
- Full ML + LLM hybrid pipeline
- Active learning with LLM as weak labeler
- Cost-optimized multi-model strategy (mini for simple, 4o for complex)
- Evaluation framework for LLM labeling quality
