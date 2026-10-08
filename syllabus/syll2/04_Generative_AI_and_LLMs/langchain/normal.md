# LangChain — Complete Roadmap / Syllabus

---

## 1. Introduction to LangChain

### 1.1 What is LangChain?
- Open-source framework for building LLM-powered applications
- Provides abstractions for chains, agents, memory, and retrieval
- Connects LLMs to external tools, data sources, and APIs
- Supports multiple LLM providers — OpenAI, Anthropic, Google, Ollama, HuggingFace
- Two main packages
  - `langchain` — core abstractions and chains
  - `langchain-community` — integrations (vector stores, tools, loaders)
  - `langchain-openai` — OpenAI-specific integration
  - `langgraph` — stateful agent graphs (separate)

### 1.2 Why LangChain?
- Eliminates boilerplate — prompt templates, output parsing, retry logic
- Composable — chain components together cleanly
- Rich integrations — 100+ vector stores, 50+ LLM providers, 100+ document loaders
- LCEL (LangChain Expression Language) — declarative, composable chain syntax
- LangSmith — built-in observability and tracing

### 1.3 Installing and Setup
- `pip install langchain langchain-openai langchain-community`
- `pip install langchain-chroma faiss-cpu` — vector stores
- Setting API key — `os.environ['OPENAI_API_KEY'] = 'sk-...'`
- Import conventions
  - `from langchain_openai import ChatOpenAI, OpenAIEmbeddings`
  - `from langchain_core.prompts import ChatPromptTemplate, PromptTemplate`
  - `from langchain_core.output_parsers import StrOutputParser, JsonOutputParser`
  - `from langchain_core.messages import HumanMessage, SystemMessage, AIMessage`
  - `from langchain.chains import LLMChain, ConversationalRetrievalChain`

### 1.4 LangChain Architecture
- **LLMs / Chat Models** — the language model interface
- **Prompt Templates** — parameterized prompts
- **Output Parsers** — structured output from LLM responses
- **Chains** — sequences of components
- **Agents** — LLMs that decide which tools to use
- **Memory** — conversation history management
- **Retrievers** — document retrieval from vector stores
- **Document Loaders** — ingesting text from files, URLs, databases
- **Text Splitters** — chunking documents for vector storage

---

## 2. Language Models

### 2.1 Chat Models (Recommended)
- `ChatOpenAI(model='gpt-4o-mini', temperature=0)`
- `ChatOpenAI(model='gpt-4o', temperature=0.7, max_tokens=500)`
- `.invoke(messages)` — synchronous call
- `.stream(messages)` — streaming
- `.batch(list_of_messages)` — parallel batch calls
- `.ainvoke(messages)` — async call

### 2.2 LLM (Legacy Text Completion)
- `OpenAI(model='gpt-3.5-turbo-instruct')` — legacy
- Takes plain string, returns plain string
- Chat models preferred for all new work

### 2.3 Other LLM Providers
- `ChatAnthropic(model='claude-3-haiku-20240307')`
- `ChatGoogleGenerativeAI(model='gemini-1.5-flash')`
- `ChatOllama(model='llama3')` — local model via Ollama
- `HuggingFaceEndpoint(repo_id='...')` — HuggingFace inference

### 2.4 Model Configuration
- `temperature` — 0 for deterministic, 0.7 for creative
- `max_tokens` — output length limit
- `top_p` — nucleus sampling
- `model_kwargs` — pass extra params
- `cache=True` — enable response caching

### 2.5 Invoking Models Directly
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

messages = [
    SystemMessage(content='You are a helpful assistant.'),
    HumanMessage(content='What is machine learning?')
]

response = llm.invoke(messages)
print(response.content)
print(response.usage_metadata)  # token usage
```

---

## 3. Prompt Templates

### 3.1 PromptTemplate (String)
```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    'Classify the sentiment of this review: {review}\nSentiment:'
)
prompt = template.format(review='This product is amazing!')
```

### 3.2 ChatPromptTemplate (Multi-turn)
```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ('system', 'You are an expert {domain} analyst.'),
    ('user', 'Analyze the following: {text}')
])

prompt = template.invoke({'domain': 'sentiment', 'text': 'Great product!'})
```

### 3.3 MessagesPlaceholder
```python
from langchain_core.prompts import MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant.'),
    MessagesPlaceholder(variable_name='history'),
    ('user', '{input}')
])
```

### 3.4 FewShotPromptTemplate
```python
from langchain_core.prompts import FewShotPromptTemplate

examples = [
    {'review': 'Great product!', 'sentiment': 'positive'},
    {'review': 'Terrible quality.', 'sentiment': 'negative'}
]
example_prompt = PromptTemplate.from_template('Review: {review}\nSentiment: {sentiment}')

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix='Classify sentiment:',
    suffix='Review: {input}\nSentiment:',
    input_variables=['input']
)
```

### 3.5 Partial Prompts
- `template.partial(domain='ML')` — pre-fill some variables
- Useful when some variables are known at chain creation time

---

## 4. Output Parsers

### 4.1 StrOutputParser
- `StrOutputParser()` — converts AIMessage to plain string
- Most common parser for general use

```python
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
result = parser.invoke(llm_response)  # returns string
```

### 4.2 JsonOutputParser
```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
# Works with model prompted to return JSON
chain = prompt | llm | parser
result = chain.invoke({'text': 'Extract entities from: John lives in Paris.'})
# Returns dict
```

### 4.3 PydanticOutputParser
```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str
    score: float
    reason: str

parser = PydanticOutputParser(pydantic_object=Sentiment)
# parser.get_format_instructions() — adds format instruction to prompt
format_instructions = parser.get_format_instructions()
```

### 4.4 CommaSeparatedListOutputParser
```python
from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
# Input: "apple, banana, cherry" → ['apple', 'banana', 'cherry']
```

### 4.5 StructuredOutputParser
- Define schema as ResponseSchema list
- `StructuredOutputParser.from_response_schemas([...])`
- Gets format instructions to inject into prompt

### 4.6 XMLOutputParser
- `XMLOutputParser()` — parse XML-formatted LLM outputs

---

## 5. LCEL — LangChain Expression Language

### 5.1 What is LCEL?
- Declarative composition syntax using `|` pipe operator
- `chain = prompt | llm | parser`
- All components are `Runnable` — same interface: `.invoke()`, `.stream()`, `.batch()`
- Composable — build complex chains from simple pieces

### 5.2 Basic Chain
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant.'),
    ('user', '{question}')
])
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
parser = StrOutputParser()

chain = prompt | llm | parser
result = chain.invoke({'question': 'What is gradient descent?'})
```

### 5.3 RunnablePassthrough
```python
from langchain_core.runnables import RunnablePassthrough

# Pass input through unchanged alongside other operations
chain = {'input': RunnablePassthrough()} | prompt | llm | parser
```

### 5.4 RunnableLambda
```python
from langchain_core.runnables import RunnableLambda

# Wrap any Python function as a Runnable
preprocess = RunnableLambda(lambda x: x.strip().lower())
chain = preprocess | prompt | llm | parser
```

### 5.5 RunnableParallel
```python
from langchain_core.runnables import RunnableParallel

# Run multiple chains in parallel
parallel = RunnableParallel({
    'summary': summary_chain,
    'sentiment': sentiment_chain,
    'keywords': keyword_chain
})
results = parallel.invoke({'text': 'Some text to analyze'})
```

### 5.6 Streaming with LCEL
```python
for chunk in chain.stream({'question': 'Explain neural networks'}):
    print(chunk, end='', flush=True)
```

### 5.7 Async with LCEL
```python
result = await chain.ainvoke({'question': 'Hello'})
async for chunk in chain.astream({'question': 'Hello'}):
    print(chunk, end='')
```

### 5.8 Batch Processing
```python
results = chain.batch([
    {'question': 'What is ML?'},
    {'question': 'What is DL?'},
    {'question': 'What is NLP?'}
])
```

---

## 6. Memory

### 6.1 What is Memory?
- Storing and retrieving conversation history
- LLMs are stateless — memory adds state
- Types — full history, summarized, entity-based, windowed

### 6.2 ChatMessageHistory
```python
from langchain_community.chat_message_histories import ChatMessageHistory

history = ChatMessageHistory()
history.add_user_message('Hello!')
history.add_ai_message('Hi! How can I help?')
history.messages  # list of messages
```

### 6.3 InMemoryChatMessageHistory
```python
from langchain_core.chat_history import InMemoryChatMessageHistory

store = {}  # session_id → history

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
```

### 6.4 RunnableWithMessageHistory
```python
from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_history,
    input_messages_key='input',
    history_messages_key='history'
)

response = chain_with_history.invoke(
    {'input': 'What is machine learning?'},
    config={'configurable': {'session_id': 'user_123'}}
)
```

### 6.5 Redis Message History
- `from langchain_community.chat_message_histories import RedisChatMessageHistory`
- `RedisChatMessageHistory(session_id=sid, url='redis://localhost:6379')`
- Persistent storage across server restarts

---

## 7. Document Loaders

### 7.1 Text and File Loaders
- `TextLoader('file.txt')` — plain text
- `PyPDFLoader('file.pdf')` — PDF pages
- `UnstructuredFileLoader('file.docx')` — Word docs, HTML, images
- `CSVLoader('data.csv')` — CSV rows as documents
- `JSONLoader('data.json', jq_schema='.[]')` — JSON with jq parsing
- `loader.load()` → list of `Document` objects
- `Document` — `.page_content` + `.metadata`

### 7.2 Web Loaders
- `WebBaseLoader('https://example.com')` — webpage content
- `WebBaseLoader(['url1', 'url2'])` — multiple pages
- `RecursiveUrlLoader('https://docs.example.com', max_depth=2)` — crawl site
- `GitLoader(clone_url='...', branch='main')` — GitHub repos
- `WikipediaLoader(query='machine learning', load_max_docs=3)` — Wikipedia

### 7.3 Directory Loader
```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader('./docs/', glob='**/*.md', loader_cls=TextLoader)
docs = loader.load()
```

### 7.4 Lazy Loading
- `loader.lazy_load()` — generator, memory-efficient for large datasets
- `loader.load_and_split(text_splitter)` — load and split in one call

---

## 8. Text Splitters

### 8.1 RecursiveCharacterTextSplitter (Recommended)
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=['\n\n', '\n', '.', ' ', '']
)
chunks = splitter.split_documents(docs)
```

### 8.2 CharacterTextSplitter
- Splits on a single separator
- `CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=50)`

### 8.3 TokenTextSplitter
- `TokenTextSplitter(chunk_size=500, chunk_overlap=50)` — split by token count
- Use when tokens matter more than characters
- Requires `tiktoken`

### 8.4 MarkdownHeaderTextSplitter
```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers = [('#', 'h1'), ('##', 'h2'), ('###', 'h3')]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(markdown_text)
# Each chunk includes header metadata
```

### 8.5 SemanticChunker
- `SemanticChunker(embeddings=OpenAIEmbeddings())` — split at semantic boundaries
- Groups semantically similar sentences together
- Higher quality chunks for RAG

---

## 9. Vector Stores

### 9.1 Chroma
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Create from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory='./chroma_db'
)

# Load existing
vectorstore = Chroma(
    persist_directory='./chroma_db',
    embedding_function=embeddings
)
```

### 9.2 FAISS
```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local('./faiss_index')

# Load
vectorstore = FAISS.load_local('./faiss_index', embeddings)
```

### 9.3 Similarity Search
```python
# Basic similarity search
results = vectorstore.similarity_search('What is gradient boosting?', k=4)

# With relevance scores
results = vectorstore.similarity_search_with_relevance_scores('query', k=4)

# MMR search — diverse results
results = vectorstore.max_marginal_relevance_search('query', k=4, fetch_k=20)
```

### 9.4 As Retriever
```python
retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 4}
)
# Or MMR
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 4, 'fetch_k': 20}
)
docs = retriever.invoke('What is a transformer?')
```

---

## 10. Retrievers

### 10.1 VectorStore Retriever
- `vectorstore.as_retriever()` — basic vector retrieval
- `search_type` — `'similarity'`, `'mmr'`, `'similarity_score_threshold'`
- `search_kwargs` — `{'k': 4, 'score_threshold': 0.7}`

### 10.2 MultiQueryRetriever
```python
from langchain.retrievers import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=ChatOpenAI(model='gpt-4o-mini')
)
# Generates multiple query variants → retrieves for each → deduplicates
```

### 10.3 ContextualCompressionRetriever
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)
# Compresses retrieved documents to only relevant parts
```

### 10.4 SelfQueryRetriever
- Generates metadata filters from natural language
- `SelfQueryRetriever.from_llm(llm, vectorstore, doc_contents, metadata_field_info)`
- `'Find documents about Python published after 2022'` → metadata filter automatically

### 10.5 EnsembleRetriever
```python
from langchain.retrievers import EnsembleRetriever

ensemble = EnsembleRetriever(
    retrievers=[sparse_retriever, dense_retriever],
    weights=[0.3, 0.7]
)
# Combines BM25 (sparse) + vector (dense) retrieval — hybrid search
```

---

## 11. RAG Chains

### 11.1 Simple RAG Chain (LCEL)
```python
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Answer using ONLY the context below.\n\nContext:\n{context}'),
    ('user', '{question}')
])

rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke('What is gradient boosting?')
```

### 11.2 RAG with Sources
```python
from langchain_core.runnables import RunnableMap

rag_chain_with_sources = RunnableMap({
    'context': retriever,
    'question': RunnablePassthrough()
}) | {
    'answer': rag_prompt | llm | StrOutputParser(),
    'sources': lambda x: [doc.metadata for doc in x['context']]
}
```

### 11.3 Conversational RAG
- `ConversationalRetrievalChain.from_llm(llm, retriever)` — legacy
- LCEL approach — `RunnableWithMessageHistory` + `rag_chain`

---

## 12. Agents

### 12.1 What are LangChain Agents?
- LLM decides which tools to call and in what order
- Agent loop — observe → think → act → observe → ...
- Tools defined as functions with descriptions
- `create_react_agent`, `create_openai_tools_agent`, `AgentExecutor`

### 12.2 Defining Tools
```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for current information about a topic."""
    return web_search_api(query)

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))
```

### 12.3 OpenAI Tools Agent
```python
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant. Use tools when needed.'),
    MessagesPlaceholder('chat_history', optional=True),
    ('human', '{input}'),
    MessagesPlaceholder('agent_scratchpad')
])

agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)
result = executor.invoke({'input': 'What is the weather in Paris and what is 25 * 4?'})
```

### 12.4 ReAct Agent
```python
from langchain.agents import create_react_agent

agent = create_react_agent(llm, tools, react_prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### 12.5 AgentExecutor Parameters
- `max_iterations` — maximum tool call iterations (default: 15)
- `max_execution_time` — time limit in seconds
- `handle_parsing_errors=True` — recover from output parse errors
- `return_intermediate_steps=True` — return tool call history
- `verbose=True` — print agent reasoning steps

### 12.6 Built-in Tools
- `TavilySearchResults(max_results=5)` — web search
- `WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())` — Wikipedia
- `PythonREPLTool()` — Python code execution (sandbox)
- `DuckDuckGoSearchRun()` — free web search
- `ArxivQueryRun()` — academic papers
- `ShellTool()` — shell commands (use with caution)

---

## 13. Chains (Legacy and Modern)

### 13.1 LLMChain (Legacy)
```python
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.invoke({'input': 'Hello'})
```

### 13.2 Sequential Chains
- `SimpleSequentialChain` — single input/output chain
- `SequentialChain` — multiple inputs/outputs
- LCEL preferred — `chain1 | chain2 | chain3`

### 13.3 Summarization Chains
```python
from langchain.chains.summarize import load_summarize_chain

chain = load_summarize_chain(llm, chain_type='map_reduce')
summary = chain.invoke({'input_documents': docs})
```

- `chain_type='stuff'` — stuff all docs into prompt (small docs)
- `chain_type='map_reduce'` — summarize each → combine (large docs)
- `chain_type='refine'` — iteratively refine summary

### 13.4 QA Chains (Legacy)
```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=True
)
```

---

## 14. LangSmith (Observability)

### 14.1 What is LangSmith?
- Observability platform for LangChain applications
- Traces every LLM call, tool use, and chain step
- View inputs, outputs, latency, token usage per step
- Debug failures and optimize prompts

### 14.2 Setup
```python
import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = 'ls-...'
os.environ['LANGCHAIN_PROJECT'] = 'my-project'
# Now all LangChain calls are automatically traced
```

### 14.3 What LangSmith Tracks
- Chain traces — step-by-step execution
- LLM inputs and outputs
- Token usage and cost per call
- Latency per step
- Error traces with full context
- Dataset creation for eval

---

## 15. Callbacks

### 15.1 StdOutCallbackHandler
```python
from langchain.callbacks import StdOutCallbackHandler

handler = StdOutCallbackHandler()
chain.invoke({'input': 'Hello'}, config={'callbacks': [handler]})
```

### 15.2 Custom Callbacks
```python
from langchain_core.callbacks import BaseCallbackHandler

class MyHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f'LLM called with: {prompts}')
    def on_llm_end(self, response, **kwargs):
        print(f'LLM returned: {response}')
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f'Tool called: {serialized["name"]}')
```

---

## 16. Common Patterns and Best Practices

### 16.1 Always Use LCEL for New Code
- Legacy chains (`LLMChain`, `SequentialChain`) are deprecated
- LCEL is composable, streamable, and async-native
- `chain = prompt | llm | parser` — always preferred

### 16.2 Error Handling
```python
from langchain_core.runnables import RunnableLambda

def safe_parse(output):
    try:
        return json.loads(output.content)
    except json.JSONDecodeError:
        return {'error': 'parse_failed', 'raw': output.content}

chain = prompt | llm | RunnableLambda(safe_parse)
```

### 16.3 Caching
```python
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())  # or SQLiteCache('cache.db')
# Repeated identical calls return cached response
```

### 16.4 Common Mistakes
- Using deprecated chain classes — use LCEL
- Not setting `max_iterations` in AgentExecutor — agents can loop forever
- Using `similarity` retrieval only — try MMR for diversity
- Not handling JSON parse failures — always validate output parser results
- Not using LangSmith — debugging chains without traces is very hard

---

## 17. Interview Topics

### 17.1 Frequently Asked Questions
- What is LCEL and why was it introduced?
- What is the difference between a Chain and an Agent?
- How does memory work in LangChain?
- What is the difference between `stuff`, `map_reduce`, and `refine` for summarization?
- How do you add tools to a LangChain agent?
- What is a retriever and how is it different from a vector store?

### 17.2 Coding Problems
- Build a simple RAG chain with LCEL using Chroma and OpenAI
- Implement a multi-turn chatbot with `RunnableWithMessageHistory`
- Create a LangChain agent with two custom tools
- Build a map-reduce summarization chain for a long document
- Implement a MultiQueryRetriever for better retrieval

---

## 18. Real-World Projects

### 18.1 Beginner Projects
- Simple Q&A chain — prompt + LLM + parser
- Document summarizer — load PDF → summarize
- Sentiment classifier with few-shot prompt template

### 18.2 Intermediate Projects
- RAG chatbot — load docs → Chroma → retriever → RAG chain + memory
- Web research agent — search tool + LLM + AgentExecutor
- Multi-document summarizer with map-reduce

### 18.3 Advanced Projects
- Customer support bot — conversational RAG + memory + streaming
- Multi-tool research agent with LangSmith tracing
- Full RAG pipeline — directory loader → splitter → Chroma → retriever → chain with sources
