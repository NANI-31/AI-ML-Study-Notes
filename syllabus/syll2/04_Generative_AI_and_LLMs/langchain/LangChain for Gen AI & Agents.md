# LangChain for Gen AI & Agents — Complete Syllabus / Roadmap

---

## 1. Introduction — LangChain as the Gen AI Application Framework

### 1.1 LangChain's Role in Gen AI
- Primary framework for building production Gen AI applications
- Provides building blocks — chains, agents, memory, retrievers, tools
- Abstracts provider differences — same code works with OpenAI, Anthropic, Gemini
- LCEL — declarative, composable, streamable chain construction
- LangGraph — stateful agent orchestration (advanced agents)
- LangSmith — tracing, debugging, and evaluation for Gen AI apps

### 1.2 Gen AI Application Types Built with LangChain
- Chatbots with memory and personality
- RAG systems — document Q&A
- Multi-tool agents — web search, code execution, API calls
- Summarization pipelines for long documents
- Structured data extraction from unstructured sources
- Multi-agent workflows (via LangGraph)

### 1.3 Installing and Setup
- `pip install langchain langchain-openai langchain-community langchain-chroma`
- `pip install tavily-python` — web search tool
- `from langchain_openai import ChatOpenAI, OpenAIEmbeddings`
- `from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder`
- `from langchain_core.runnables.history import RunnableWithMessageHistory`
- `from langchain.agents import create_openai_tools_agent, AgentExecutor`
- `from langchain.tools import tool`

---

## 2. Building Chatbots with LangChain

### 2.1 Simple Single-Turn Chain
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant. Be concise and friendly.'),
    ('user', '{input}')
])

chain = prompt | llm | StrOutputParser()
response = chain.invoke({'input': 'What is RAG in AI?'})
```

### 2.2 Multi-turn Chatbot with Memory
```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful AI assistant.'),
    MessagesPlaceholder(variable_name='history'),
    ('user', '{input}')
])

chain = prompt | llm | StrOutputParser()

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='history'
)

# First turn
r1 = chatbot.invoke(
    {'input': 'My name is Arjun.'},
    config={'configurable': {'session_id': 'session_1'}}
)

# Second turn — model remembers the name
r2 = chatbot.invoke(
    {'input': 'What is my name?'},
    config={'configurable': {'session_id': 'session_1'}}
)
```

### 2.3 Streaming Chatbot
```python
for chunk in chatbot.stream(
    {'input': 'Explain neural networks step by step.'},
    config={'configurable': {'session_id': 'session_1'}}
):
    print(chunk, end='', flush=True)
```

### 2.4 Trimming Long Histories
```python
from langchain_core.messages import trim_messages

trimmer = trim_messages(
    max_tokens=2000,
    strategy='last',
    token_counter=llm,
    include_system=True,
    allow_partial=False
)

# Use in chain
chain = (
    RunnablePassthrough.assign(history=lambda x: trimmer.invoke(x['history']))
    | prompt | llm | StrOutputParser()
)
```

---

## 3. RAG Applications with LangChain

### 3.1 Full RAG Pipeline
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough

# Step 1: Load
loader = PyPDFLoader('knowledge_base.pdf')
docs = loader.load()

# Step 2: Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Step 3: Embed and Store
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory='./db')

# Step 4: Retrieve
retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

# Step 5: Generate
def format_docs(docs):
    return '\n\n'.join(d.page_content for d in docs)

rag_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Answer using ONLY the context below. If unsure, say "I don\'t know."\n\nContext:\n{context}'),
    ('user', '{question}')
])

rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | rag_prompt | llm | StrOutputParser()
)

answer = rag_chain.invoke('What is the refund policy?')
```

### 3.2 RAG with Sources
```python
from langchain_core.runnables import RunnableMap

rag_with_sources = RunnableMap({
    'docs': retriever,
    'question': RunnablePassthrough()
}) | {
    'answer': (
        lambda x: {'context': format_docs(x['docs']), 'question': x['question']}
    ) | rag_prompt | llm | StrOutputParser(),
    'sources': lambda x: [
        {'source': d.metadata.get('source', 'unknown'), 'page': d.metadata.get('page', 0)}
        for d in x['docs']
    ]
}

result = rag_with_sources.invoke('What are the pricing tiers?')
print(result['answer'])
print(result['sources'])
```

### 3.3 Conversational RAG
```python
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Step 1: Rephrase question using conversation history
contextualize_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Rephrase the user question as a standalone question, given the conversation history. Return ONLY the rephrased question.'),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

# Step 2: Answer with retrieved context
qa_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Answer using only the context below.\n\nContext:\n{context}'),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
])
qa_chain = create_stuff_documents_chain(llm, qa_prompt)

# Step 3: Full conversational RAG chain
conv_rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# With memory
conv_rag_with_history = RunnableWithMessageHistory(
    conv_rag_chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)
```

### 3.4 Advanced Retrieval Strategies
```python
from langchain.retrievers import MultiQueryRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# MultiQuery — generate multiple phrasings
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={'k': 4}),
    llm=llm
)

# Hybrid — BM25 (sparse) + dense vector
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vectorstore.as_retriever(search_kwargs={'k': 4})],
    weights=[0.3, 0.7]
)

# MMR — maximal marginal relevance (diverse results)
mmr_retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 4, 'fetch_k': 20}
)
```

---

## 4. Building Agents with LangChain

### 4.1 Custom Tool Definition
```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for real-time information. Use for current events, prices, news."""
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.environ['TAVILY_API_KEY'])
    results = client.search(query, max_results=3)
    return '\n'.join(r['content'] for r in results['results'])

@tool
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression. Input: valid Python math expression."""
    import ast, operator
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
    try:
        tree = ast.parse(expression, mode='eval')
        result = eval(compile(tree, '', 'eval'))
        return str(result)
    except Exception as e:
        return f'Error: {e}'

@tool
def get_current_date() -> str:
    """Returns today's date in YYYY-MM-DD format."""
    from datetime import date
    return str(date.today())
```

### 4.2 OpenAI Tools Agent
```python
from langchain.agents import create_openai_tools_agent, AgentExecutor

tools = [search_web, calculate, get_current_date]

agent_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant with access to tools. '
               'Use tools when you need real-time information or need to compute something. '
               'Always explain what you are doing before using a tool.'),
    MessagesPlaceholder('chat_history', optional=True),
    ('human', '{input}'),
    MessagesPlaceholder('agent_scratchpad')
])

agent = create_openai_tools_agent(llm, tools, agent_prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

result = executor.invoke({'input': 'What is today\'s date and what is 15% of 350?'})
print(result['output'])
print(result['intermediate_steps'])  # tool calls and results
```

### 4.3 Agent with Memory
```python
agent_with_memory = RunnableWithMessageHistory(
    executor,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)

# Turn 1
r1 = agent_with_memory.invoke(
    {'input': 'Search for the latest news about AI.'},
    config={'configurable': {'session_id': 'agent_session_1'}}
)

# Turn 2 — agent remembers previous search
r2 = agent_with_memory.invoke(
    {'input': 'Summarize what you found.'},
    config={'configurable': {'session_id': 'agent_session_1'}}
)
```

### 4.4 ReAct Agent
```python
from langchain.agents import create_react_agent
from langchain import hub

# Pull ReAct prompt from hub
react_prompt = hub.pull('hwchase17/react')

react_agent = create_react_agent(llm, tools, react_prompt)
react_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)
```

### 4.5 Built-in Agent Tools
```python
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool

tavily = TavilySearchResults(max_results=5)
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
python_repl = PythonREPLTool()

tools = [tavily, wikipedia, python_repl]
```

---

## 5. Structured Outputs in Gen AI

### 5.1 .with_structured_output()
```python
from pydantic import BaseModel
from typing import Literal

class RoutingDecision(BaseModel):
    category: Literal['billing', 'technical', 'sales', 'general']
    confidence: Literal['high', 'medium', 'low']
    suggested_action: str

structured_llm = llm.with_structured_output(RoutingDecision)

result = structured_llm.invoke(
    'Route this message: "My payment was declined yesterday"'
)
print(result.category)  # 'billing'
print(result.confidence)  # 'high'
```

### 5.2 JSON Schema Output
```python
schema = {
    'title': 'Entity',
    'type': 'object',
    'properties': {
        'names': {'type': 'array', 'items': {'type': 'string'}},
        'locations': {'type': 'array', 'items': {'type': 'string'}}
    }
}
structured_llm = llm.with_structured_output(schema)
```

### 5.3 Chaining Structured Output
```python
classify_chain = (
    ChatPromptTemplate.from_messages([('user', 'Classify: {input}')])
    | llm.with_structured_output(RoutingDecision)
)

result = classify_chain.invoke({'input': 'I need help with my invoice'})
```

---

## 6. Summarization Chains

### 6.1 Stuff (Small Documents)
```python
from langchain.chains.summarize import load_summarize_chain

chain = load_summarize_chain(llm, chain_type='stuff')
summary = chain.invoke({'input_documents': docs})['output_text']
```

### 6.2 Map-Reduce (Large Documents)
```python
from langchain_core.prompts import PromptTemplate

map_prompt = PromptTemplate.from_template(
    'Summarize this section in 3 bullet points:\n\n{text}'
)
combine_prompt = PromptTemplate.from_template(
    'Write a comprehensive summary from these section summaries:\n\n{text}'
)

chain = load_summarize_chain(
    llm,
    chain_type='map_reduce',
    map_prompt=map_prompt,
    combine_prompt=combine_prompt,
    verbose=True
)
summary = chain.invoke({'input_documents': chunks})['output_text']
```

### 6.3 Custom Map-Reduce with LCEL
```python
# Map step
map_chain = map_prompt | llm | StrOutputParser()

# Process each chunk
chunk_summaries = map_chain.batch([{'text': c.page_content} for c in chunks])

# Reduce step
combined_text = '\n\n'.join(chunk_summaries)
final_summary = (combine_prompt | llm | StrOutputParser()).invoke({'text': combined_text})
```

---

## 7. Tool Calling Deep Dive

### 7.1 Tool with Complex Return Type
```python
@tool
def analyze_csv(filepath: str, question: str) -> str:
    """
    Analyze a CSV file to answer data questions.
    Args:
        filepath: Path to the CSV file
        question: Question to answer about the data
    """
    import pandas as pd
    df = pd.read_csv(filepath)
    context = f'Columns: {df.columns.tolist()}\nShape: {df.shape}\nSample:\n{df.head(3).to_string()}'
    analysis_chain = (
        ChatPromptTemplate.from_messages([('user', f'Data info:\n{context}\nQuestion: {question}')])
        | ChatOpenAI(model='gpt-4o-mini', temperature=0)
        | StrOutputParser()
    )
    return analysis_chain.invoke({})
```

### 7.2 Inspecting Tool Calls
```python
result = executor.invoke(
    {'input': 'Search for Python best practices'},
    return_only_outputs=False
)

for step in result['intermediate_steps']:
    action, observation = step
    print(f'Tool: {action.tool}')
    print(f'Input: {action.tool_input}')
    print(f'Output: {observation[:200]}')
    print('---')
```

### 7.3 Tool Error Handling
```python
@tool
def safe_api_call(endpoint: str) -> str:
    """Call an external API endpoint."""
    try:
        import requests
        response = requests.get(endpoint, timeout=5)
        response.raise_for_status()
        return response.text[:1000]
    except Exception as e:
        return f'API call failed: {str(e)}. Try a different endpoint.'
```

---

## 8. LangChain Callbacks for Gen AI

### 8.1 Streaming Callback
```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm_streaming = ChatOpenAI(
    model='gpt-4o-mini',
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
chain_streaming = prompt | llm_streaming
chain_streaming.invoke({'input': 'Tell me about LangChain'})
```

### 8.2 Usage Tracking Callback
```python
from langchain_core.callbacks import BaseCallbackHandler

class UsageTracker(BaseCallbackHandler):
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0

    def on_llm_end(self, response, **kwargs):
        if response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            tokens = usage.get('total_tokens', 0)
            self.total_tokens += tokens
            self.total_cost += tokens * 0.00000015  # approx gpt-4o-mini cost

tracker = UsageTracker()
chain.invoke({'input': 'Hello'}, config={'callbacks': [tracker]})
print(f'Tokens used: {tracker.total_tokens}, Cost: ${tracker.total_cost:.6f}')
```

---

## 9. FastAPI Integration for Gen AI Apps

### 9.1 Basic FastAPI + LangChain Endpoint
```python
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()
llm = ChatOpenAI(model='gpt-4o-mini')
chain = ChatPromptTemplate.from_messages([('user', '{message}')]) | llm | StrOutputParser()

class ChatRequest(BaseModel):
    message: str
    session_id: str = 'default'

@app.post('/chat')
async def chat(req: ChatRequest):
    response = await chain.ainvoke({'message': req.message})
    return {'response': response}
```

### 9.2 Streaming Endpoint
```python
from fastapi.responses import StreamingResponse

@app.post('/chat/stream')
async def chat_stream(req: ChatRequest):
    async def generate():
        async for chunk in chain.astream({'message': req.message}):
            yield f'data: {chunk}\n\n'

    return StreamingResponse(generate(), media_type='text/event-stream')
```

---

## 10. Production Patterns for Gen AI with LangChain

### 10.1 LangSmith Tracing in Production
```python
import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = 'ls-...'
os.environ['LANGCHAIN_PROJECT'] = 'production-chatbot'

# Add run name for easier filtering
result = chain.invoke(
    {'input': user_message},
    config={'run_name': f'chat_{session_id}', 'tags': ['production', 'chatbot']}
)
```

### 10.2 Retry Logic
```python
from langchain_core.runnables import RunnableRetry

chain_with_retry = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)
```

### 10.3 Fallback Models
```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

primary = ChatOpenAI(model='gpt-4o-mini')
fallback = ChatAnthropic(model='claude-3-haiku-20240307')

llm_with_fallback = primary.with_fallbacks([fallback])
chain = prompt | llm_with_fallback | StrOutputParser()
```

### 10.4 Request Timeout
```python
llm = ChatOpenAI(model='gpt-4o-mini', timeout=30)
chain_with_timeout = chain.with_config({'timeout': 30})
```

---

## 11. Evaluation of Gen AI Apps with LangChain

### 11.1 LangChain Evaluators
```python
from langchain.evaluation import load_evaluator

# String correctness judge
correctness = load_evaluator('labeled_criteria', criteria='correctness', llm=llm)
helpfulness = load_evaluator('criteria', criteria='helpfulness', llm=llm)

result = correctness.evaluate_strings(
    prediction='The capital of France is Paris.',
    reference='Paris',
    input='What is the capital of France?'
)
```

### 11.2 RAG-specific Evaluation
```python
# Faithfulness check — does answer come from context?
faithfulness_eval = load_evaluator(
    'labeled_criteria',
    criteria={'faithfulness': 'Is the answer supported by the context?'},
    llm=llm
)
```

### 11.3 Building Custom Eval Pipeline
```python
eval_cases = [
    {'input': 'What is the refund policy?', 'expected': 'within 30 days', 'context': docs},
    {'input': 'What are the payment methods?', 'expected': 'card and bank transfer', 'context': docs}
]

results = []
for case in eval_cases:
    answer = rag_chain.invoke(case['input'])
    score = correctness.evaluate_strings(
        prediction=answer,
        reference=case['expected'],
        input=case['input']
    )
    results.append({'input': case['input'], 'answer': answer, 'score': score['score']})

accuracy = sum(r['score'] for r in results) / len(results)
print(f'RAG accuracy: {accuracy:.2%}')
```

---

## 12. LangChain Hub

### 12.1 Using Prompts from Hub
```python
from langchain import hub

# Pull community prompts
react_prompt = hub.pull('hwchase17/react')
rag_prompt = hub.pull('rlm/rag-prompt')

# Use in chain
chain = rag_prompt | llm | StrOutputParser()
```

### 12.2 Pushing Prompts to Hub
```python
hub.push('my-org/my-custom-prompt', my_prompt, new_repo_is_public=False)
```

---

## 13. Interview Topics for Gen AI Engineers

### 13.1 Frequently Asked Questions
- What is LCEL and how does it differ from legacy LangChain chains?
- How does `RunnableWithMessageHistory` manage conversation state?
- What is the difference between a chain and an agent in LangChain?
- How do you implement conversational RAG in LangChain?
- How do you handle streaming in a LangChain application?
- What is `create_history_aware_retriever` and why is it needed?
- How do you add fallbacks and retries to a LangChain chain?

### 13.2 Gen AI Coding Problems
- Build a multi-turn chatbot with `RunnableWithMessageHistory`
- Implement a full RAG pipeline with sources and streaming
- Create an agent with 3 custom tools and `AgentExecutor`
- Build a conversational RAG system that rephrases questions
- Implement a map-reduce summarization chain for a long PDF

---

## 14. Real-World Gen AI Projects

### 14.1 Beginner Projects
- CLI chatbot with multi-turn memory
- PDF Q&A chatbot — load → split → embed → RAG chain
- Intent router — classify user input, return different responses

### 14.2 Intermediate Projects
- Customer support bot — RAG + memory + streaming + tool calling
- Research agent — web search + Wikipedia + calculator + synthesis
- Conversational RAG over company docs

### 14.3 Advanced Projects
- Multi-tool agent shipped as FastAPI `/chat` endpoint with streaming
- Conversational RAG + LangSmith eval pipeline
- Agent with fallback models and retry logic deployed to Render/HF Spaces

---

## 15. Roadmap

### 15.1 Beginner Level
- LCEL composition — `prompt | llm | parser`
- `ChatPromptTemplate` + `MessagesPlaceholder`
- `InMemoryChatMessageHistory` + `RunnableWithMessageHistory`
- Simple RAG — loader → splitter → Chroma → retriever → chain

### 15.2 Intermediate Level
- Conversational RAG with `create_history_aware_retriever`
- Custom tools with `@tool` decorator
- `AgentExecutor` with `create_openai_tools_agent`
- Structured output with `.with_structured_output()`
- LangSmith tracing setup

### 15.3 Advanced Level
- Multi-tool agents with error handling and max iterations
- RAG with hybrid retrieval (BM25 + dense)
- FastAPI integration with streaming SSE
- Fallback chains and retry logic
- Eval pipeline with LangChain evaluators
