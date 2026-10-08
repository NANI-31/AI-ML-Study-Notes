# LangChain for Machine Learning — Complete Syllabus / Roadmap

---

## 1. Introduction — LangChain in the ML Workflow

### 1.1 Where LangChain Fits in ML Pipelines
- LangChain as orchestration layer on top of OpenAI API
- Reduces boilerplate — templating, parsing, chaining handled automatically
- Useful in ML for — data labeling pipelines, feature extraction, model explanation
- Enables prompt versioning, caching, and tracing in ML pipelines
- Not replacing sklearn/XGBoost — augmenting them with LLM capabilities

### 1.2 High-Value LangChain Use Cases in ML
- Structured feature extraction from text using output parsers
- Batch labeling pipelines with retry and error handling
- LLM-generated model explanations with structured output
- Automated EDA report generation
- Document Q&A for ML documentation and notebooks

### 1.3 Installing and Setup
- `pip install langchain langchain-openai langchain-chroma`
- `from langchain_openai import ChatOpenAI, OpenAIEmbeddings`
- `from langchain_core.prompts import ChatPromptTemplate`
- `from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser`
- `from langchain_core.runnables import RunnableLambda, RunnablePassthrough`

---

## 2. Structured Feature Extraction with LangChain

### 2.1 PydanticOutputParser for ML Features
```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class SupportTicketFeatures(BaseModel):
    category: Literal['billing', 'technical', 'shipping', 'general']
    urgency: int = Field(ge=1, le=5, description='1=low, 5=critical')
    sentiment: Literal['positive', 'negative', 'neutral']
    has_order_number: bool
    requires_escalation: bool

parser = PydanticOutputParser(pydantic_object=SupportTicketFeatures)

prompt = ChatPromptTemplate.from_messages([
    ('system', 'Extract features from customer support tickets.\n{format_instructions}'),
    ('user', 'Ticket: {text}')
]).partial(format_instructions=parser.get_format_instructions())

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

chain = prompt | llm | parser
features = chain.invoke({'text': 'My order #12345 has not arrived after 3 weeks!!!'})
print(features.category, features.urgency, features.has_order_number)
```

### 2.2 Batch Feature Extraction
```python
import pandas as pd

tickets = df['text'].tolist()
inputs = [{'text': t} for t in tickets]

# Batch invoke — parallel
raw_results = chain.batch(inputs, config={'max_concurrency': 5})

# Convert to DataFrame
feature_df = pd.DataFrame([r.model_dump() for r in raw_results if r is not None])
df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
```

### 2.3 Error Handling in Batch Extraction
```python
from langchain_core.runnables import RunnableLambda

def safe_parse(output):
    try:
        return parser.parse(output.content)
    except Exception:
        return None

safe_chain = prompt | llm | RunnableLambda(safe_parse)

results = safe_chain.batch(inputs)
valid = [r for r in results if r is not None]
failed_count = len(results) - len(valid)
print(f'Extracted: {len(valid)}, Failed: {failed_count}')
```

---

## 3. LangChain Prompt Templates for ML Tasks

### 3.1 Parameterized Labeling Template
```python
from langchain_core.prompts import PromptTemplate

labeling_template = PromptTemplate.from_template(
    'You are a data labeling expert.\n'
    'Task: {task_description}\n'
    'Categories: {categories}\n'
    'Return ONLY one of the listed categories.\n\n'
    'Input: {text}\n'
    'Category:'
)

chain = labeling_template | llm | StrOutputParser()

# Reuse template for different labeling tasks
sentiment_result = chain.invoke({
    'task_description': 'Classify customer sentiment',
    'categories': 'positive, negative, neutral',
    'text': 'Great product! Will buy again.'
})
```

### 3.2 Few-Shot Labeling Template
```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {'text': 'My payment was charged twice', 'category': 'billing'},
    {'text': 'The app crashes on startup', 'category': 'technical'},
    {'text': 'Where is my shipment?', 'category': 'shipping'}
]
example_prompt = PromptTemplate.from_template('Text: {text}\nCategory: {category}')

few_shot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix='Classify support tickets:',
    suffix='Text: {input}\nCategory:',
    input_variables=['input']
)

chain = few_shot | llm | StrOutputParser()
```

### 3.3 Dynamic Few-Shot with SemanticSimilarityExampleSelector
```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma

selector = SemanticSimilarityExampleSelector.from_examples(
    examples=example_pool,
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=3  # select 3 most similar examples
)

dynamic_few_shot = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    prefix='Classify:',
    suffix='Text: {input}\nCategory:',
    input_variables=['input']
)
```

---

## 4. LangChain for Data Cleaning Pipelines

### 4.1 Text Normalization Chain
```python
normalize_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Normalize the input text:\n'
               '1. Fix spelling and grammar\n'
               '2. Expand abbreviations\n'
               '3. Standardize to title case for names\n'
               'Return ONLY the corrected text.'),
    ('user', '{text}')
])

normalize_chain = normalize_prompt | llm | StrOutputParser()
df['cleaned_text'] = df['raw_text'].apply(
    lambda t: normalize_chain.invoke({'text': t})
)
```

### 4.2 Duplicate Detection Chain
```python
class DuplicateCheck(BaseModel):
    are_duplicates: bool
    confidence: Literal['high', 'medium', 'low']
    reason: str

dedup_parser = PydanticOutputParser(pydantic_object=DuplicateCheck)

dedup_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Determine if two texts refer to the same entity.\n{format_instructions}'),
    ('user', 'Text 1: {text1}\nText 2: {text2}')
]).partial(format_instructions=dedup_parser.get_format_instructions())

dedup_chain = dedup_prompt | llm | dedup_parser
```

---

## 5. LangChain for EDA and Feature Engineering

### 5.1 Automated Dataset Description
```python
import json

def describe_dataset(df):
    df_info = {
        'shape': df.shape,
        'columns': df.dtypes.astype(str).to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'sample': df.head(3).to_dict(orient='records'),
        'stats': df.describe().to_dict()
    }

    eda_prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a data scientist. Analyze the dataset and provide insights.'),
        ('user', 'Dataset info:\n{info}\n\n'
                 'Provide:\n'
                 '1. What this dataset likely represents\n'
                 '2. Top 3 data quality issues\n'
                 '3. Top 5 feature engineering suggestions\n'
                 'Be specific and actionable. Return as JSON.')
    ])

    chain = eda_prompt | llm | JsonOutputParser()
    return chain.invoke({'info': json.dumps(df_info, default=str)})
```

### 5.2 Feature Suggestion Chain
```python
class FeatureSuggestion(BaseModel):
    name: str
    formula: str
    rationale: str
    expected_importance: Literal['high', 'medium', 'low']

class FeatureSuggestions(BaseModel):
    suggestions: list[FeatureSuggestion]

parser = PydanticOutputParser(pydantic_object=FeatureSuggestions)

feature_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are an ML feature engineering expert.\n{format_instructions}'),
    ('user', 'Task: predict {target}\nCurrent features: {features}\nSuggest 5 new engineered features.')
]).partial(format_instructions=parser.get_format_instructions())

chain = feature_prompt | llm | parser
suggestions = chain.invoke({'target': 'customer churn', 'features': str(df.columns.tolist())})
```

---

## 6. LangChain for Model Explanation

### 6.1 SHAP Explanation Generator
```python
class ModelExplanation(BaseModel):
    headline: str = Field(description='One sentence summary for business audience')
    key_factors: list[str] = Field(description='Top 3 factors in plain language')
    recommended_action: str = Field(description='What the business should do')

explain_parser = PydanticOutputParser(pydantic_object=ModelExplanation)

explain_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Translate ML model predictions into business language.\n{format_instructions}'),
    ('user', 'Prediction: {prediction}\nTop SHAP features:\n{shap_table}')
]).partial(format_instructions=explain_parser.get_format_instructions())

explain_chain = explain_prompt | llm | explain_parser

def generate_explanation(prediction, shap_values, feature_names):
    shap_table = '\n'.join(
        f'{f}: {v:+.3f}' for f, v in sorted(
            zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True
        )[:5]
    )
    return explain_chain.invoke({'prediction': prediction, 'shap_table': shap_table})
```

### 6.2 Metric Report Chain
```python
report_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Write ML model performance reports for business stakeholders. Use plain language. No ML jargon.'),
    ('user', 'Model: {model_name}\nTask: {task}\nMetrics: {metrics}\n\nWrite a 3-sentence performance summary.')
])

report_chain = report_prompt | llm | StrOutputParser()

report = report_chain.invoke({
    'model_name': 'Customer Churn Classifier v2',
    'task': 'Binary classification — predict customer churn',
    'metrics': 'Accuracy: 87%, F1: 0.84, AUC: 0.91'
})
```

---

## 7. Document-Based ML Workflows

### 7.1 Loading and Processing ML Documentation
```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load research papers, documentation
loader = DirectoryLoader('./ml_papers/', glob='*.pdf', loader_cls=PyPDFLoader)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)
```

### 7.2 Building a ML Knowledge Base
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory='./ml_kb')

retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
```

### 7.3 Q&A over ML Notebooks or Documentation
```python
qa_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Answer questions about ML techniques using the provided documentation.\n'
               'Context:\n{context}\n\n'
               'If the answer is not in the context, say so.'),
    ('user', '{question}')
])

def format_docs(docs):
    return '\n\n'.join(d.page_content for d in docs)

qa_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | qa_prompt | llm | StrOutputParser()
)

answer = qa_chain.invoke('How does XGBoost handle missing values?')
```

---

## 8. Caching for ML Pipelines

### 8.1 In-Memory Cache
```python
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())
# Identical prompts return cached response — zero API cost
```

### 8.2 SQLite Cache (Persistent)
```python
from langchain.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path='.langchain_cache.db'))
# Persists across runs — great for ML preprocessing pipelines
```

### 8.3 When Caching Helps in ML
- Repeated labeling of same samples (dedup before API call)
- Re-running pipelines during development
- Template-heavy prompts with same static parts
- Test runs on same subset

---

## 9. Evaluation with LangChain

### 9.1 LangChain Evaluators
```python
from langchain.evaluation import load_evaluator

# LLM-as-judge accuracy evaluator
evaluator = load_evaluator('labeled_criteria', llm=llm, criteria='correctness')

result = evaluator.evaluate_strings(
    prediction='The answer is 42',
    reference='42',
    input='What is the answer?'
)
print(result['score'])  # 1 or 0
```

### 9.2 Embedding Distance Evaluator
```python
evaluator = load_evaluator('embedding_distance', embeddings=OpenAIEmbeddings())
result = evaluator.evaluate_strings(prediction=pred, reference=expected)
# result['score'] — cosine distance (lower = more similar)
```

### 9.3 Building an Eval Loop
```python
eval_set = [
    {'input': 'positive review text', 'expected': 'positive'},
    {'input': 'negative review text', 'expected': 'negative'}
]

scores = []
for case in eval_set:
    prediction = chain.invoke({'text': case['input']})
    score = 1 if prediction.strip().lower() == case['expected'] else 0
    scores.append(score)

print(f'Accuracy: {sum(scores)/len(scores):.2%}')
```

---

## 10. LangSmith for ML Pipeline Observability

### 10.1 Setup LangSmith
```python
import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = 'ls-...'
os.environ['LANGCHAIN_PROJECT'] = 'ml-labeling-pipeline'
```

### 10.2 What LangSmith Shows for ML Pipelines
- Every prompt sent to LLM — inspect for issues
- Every output — check for parse errors
- Token usage per labeling call
- Latency per request
- Failed runs with full context

### 10.3 Creating Datasets in LangSmith
- Upload eval set to LangSmith
- Run chain on dataset → store predictions
- Compare prompt versions on same eval set
- Track accuracy improvements over prompt iterations

---

## 11. Best Practices for LangChain in ML

### 11.1 Always Do
- Use LCEL — not legacy chain classes
- Use PydanticOutputParser — type-safe, validated output
- Set `temperature=0` — consistent labeling and extraction
- Enable caching — reduce cost during development
- Set `max_concurrency` in `.batch()` — prevent rate limit errors
- Enable LangSmith tracing — debug failures quickly

### 11.2 Never Do
- Use `LLMChain` or `SequentialChain` for new code — deprecated
- Parse raw LLM strings manually — use output parsers
- Skip error handling in batch pipelines — some calls will fail
- Send full dataset to LLM without testing on 20 samples first
- Ignore token limits — large batches can hit context window

### 11.3 Cost Control
- SQLite cache for repeated calls
- `gpt-4o-mini` for all ML data tasks
- Set `max_tokens` — extraction needs very few output tokens
- Batch API (direct OpenAI) for >1k samples — LangChain doesn't expose Batch API natively

---

## 12. Interview Topics

### 12.1 Frequently Asked Questions
- How do you use PydanticOutputParser for structured extraction?
- What is LCEL and why should you use it over LLMChain?
- How would you build a batch feature extraction pipeline with LangChain?
- How does LangSmith help debug a LangChain ML pipeline?
- How do you handle parsing failures in a batch LangChain pipeline?

### 12.2 ML-Oriented Coding Problems
- Build a structured feature extraction chain using Pydantic
- Implement a few-shot labeling chain with dynamic example selection
- Create a SHAP-to-natural-language explanation chain
- Build a batch labeling pipeline with error handling and validation
- Set up LangSmith tracing and create an eval dataset from results

---

## 13. Real-World Projects

### 13.1 Beginner Projects
- Batch sentiment labeler with PydanticOutputParser
- Automated EDA report generator from DataFrame info
- Simple few-shot text classifier using ChatPromptTemplate

### 13.2 Intermediate Projects
- Multi-field feature extraction pipeline for e-commerce text data
- SHAP + LangChain model explanation generator
- Dynamic few-shot labeler with SemanticSimilarityExampleSelector

### 13.3 Advanced Projects
- Full ML data pipeline — load → clean → extract features → validate → export
- LangSmith-tracked prompt versioning and A/B testing for labeling accuracy
- Q&A system over ML research papers with RAG

---

## 14. Roadmap

### 14.1 Beginner Level
- ChatPromptTemplate + ChatOpenAI + StrOutputParser
- PydanticOutputParser for structured extraction
- Batch invoke for parallel processing
- InMemoryCache setup

### 14.2 Intermediate Level
- FewShotPromptTemplate + SemanticSimilarityExampleSelector
- LCEL pipeline composition
- Error handling with RunnableLambda
- LangSmith tracing

### 14.3 Advanced Level
- Full extraction pipeline with validation and failure routing
- Dynamic few-shot with Chroma vector store
- RAG pipeline over ML documentation
- Eval loop with LangChain evaluators
