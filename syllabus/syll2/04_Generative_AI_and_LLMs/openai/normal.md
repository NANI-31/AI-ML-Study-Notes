# OpenAI API — Complete Roadmap / Syllabus

---

## 1. Introduction to OpenAI API

### 1.1 What is the OpenAI API?
- REST API that exposes OpenAI's models over HTTP
- Access GPT-4, GPT-4o, GPT-4o mini, o1, o3 via API calls
- Pay-per-token pricing — no model hosting required
- Official Python SDK — `openai` package
- Capabilities
  - Text generation (chat completions)
  - Embeddings
  - Image generation (DALL·E)
  - Audio transcription and speech (Whisper, TTS)
  - Fine-tuning
  - Assistants API (threads, tools, files)
  - Moderation
  - Batch API

### 1.2 Installing and Setup
- `pip install openai`
- Setting API key
  - `export OPENAI_API_KEY='sk-...'` — environment variable (recommended)
  - `from openai import OpenAI; client = OpenAI(api_key='sk-...')` — explicit
  - `.env` file with `python-dotenv` — `load_dotenv()`
- Import convention — `from openai import OpenAI`
- Creating client — `client = OpenAI()`
- Verifying — `client.models.list()`

### 1.3 Core Concepts
- **Token** — atomic unit of text (~4 chars in English)
- **Context window** — maximum tokens the model can process at once
- **Prompt** — input text / instruction
- **Completion** — model output
- **Temperature** — randomness of output (0 = deterministic, 2 = very random)
- **Top-p** — nucleus sampling probability threshold
- **System message** — persistent instruction for the model's behavior
- **User message** — the human's turn
- **Assistant message** — the model's turn

### 1.4 Available Models
- `gpt-4o` — flagship multimodal model
- `gpt-4o-mini` — fast, cost-efficient
- `gpt-4-turbo` — large context (128k)
- `o1` — advanced reasoning model
- `o3-mini` — efficient reasoning model
- `text-embedding-3-small` — compact embedding model
- `text-embedding-3-large` — high-quality embeddings
- `dall-e-3` — image generation
- `whisper-1` — audio transcription
- `tts-1`, `tts-1-hd` — text-to-speech

---

## 2. Chat Completions

### 2.1 Basic Chat Completion
```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'What is gradient boosting?'}
    ]
)
print(response.choices[0].message.content)
```

### 2.2 Message Roles
- `'system'` — sets model behavior and persona
- `'user'` — human turn input
- `'assistant'` — model response (for multi-turn conversations)
- `'tool'` — result from a tool/function call

### 2.3 Multi-turn Conversation
- Maintain `messages` list — append each turn
- `messages.append({'role': 'assistant', 'content': response_text})`
- `messages.append({'role': 'user', 'content': next_input})`
- Context window limit — truncate older messages when approaching limit

### 2.4 Response Object
- `response.choices[0].message.content` — text output
- `response.choices[0].finish_reason` — `'stop'`, `'length'`, `'tool_calls'`
- `response.usage.prompt_tokens` — input tokens
- `response.usage.completion_tokens` — output tokens
- `response.usage.total_tokens` — total
- `response.model` — model actually used
- `response.id` — unique response ID

### 2.5 Key Parameters
- `model` — model name string
- `messages` — list of message dicts
- `temperature` — 0–2, default: 1
- `max_tokens` / `max_completion_tokens` — output limit
- `top_p` — nucleus sampling (0–1)
- `n` — number of completions to generate
- `stop` — stop sequences (string or list)
- `presence_penalty` — -2 to 2, penalize new topics
- `frequency_penalty` — -2 to 2, penalize repetition
- `seed` — for reproducible outputs
- `logprobs` — return log probabilities
- `top_logprobs` — top N token probabilities per position

---

## 3. Streaming

### 3.1 Streaming Chat Completions
```python
stream = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': 'Tell me about transformers'}],
    stream=True
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end='', flush=True)
```

### 3.2 Streaming with Context Manager
- `with client.chat.completions.stream(...) as stream:` — new streaming API
- `for text in stream.text_stream:` — direct text iteration
- `stream.get_final_completion()` — full completion after stream ends

### 3.3 When to Use Streaming
- User-facing applications — show output as it generates
- Long outputs — avoid timeout
- Chat interfaces — improves perceived response time

---

## 4. Token Management

### 4.1 Counting Tokens
- Install tiktoken — `pip install tiktoken`
- `enc = tiktoken.encoding_for_model('gpt-4o')`
- `tokens = enc.encode(text)`
- `n_tokens = len(tokens)` — token count
- `enc.decode(tokens)` — tokens back to text

### 4.2 Token Costs (Approximate)
- Input tokens — cheaper than output tokens
- `gpt-4o-mini` — cheapest for most tasks
- `gpt-4o` — 10–15× more expensive than mini
- Cost = `(prompt_tokens × input_price + completion_tokens × output_price) / 1_000_000`

### 4.3 Context Window Limits
- `gpt-4o` — 128k tokens context window
- `gpt-4o-mini` — 128k tokens
- `o1` — 200k tokens
- Monitor `response.usage.prompt_tokens` in production
- Truncation strategy — keep system prompt + recent messages + current query

### 4.4 Reducing Token Usage
- Concise system prompts
- Truncate long conversation history
- Use `max_tokens` to cap output length
- Choose cheaper model for simple tasks
- Use Batch API for non-real-time workloads (50% discount)

---

## 5. Structured Outputs

### 5.1 JSON Mode
```python
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {'role': 'system', 'content': 'Return JSON only.'},
        {'role': 'user', 'content': 'Extract name and age from: John is 25 years old.'}
    ],
    response_format={'type': 'json_object'}
)
import json
data = json.loads(response.choices[0].message.content)
```

### 5.2 Structured Outputs with Pydantic (Strict JSON)
```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

response = client.beta.chat.completions.parse(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': 'John is 25 years old'}],
    response_format=Person
)
person = response.choices[0].message.parsed
print(person.name, person.age)
```

### 5.3 Structured Outputs with JSON Schema
- `response_format={'type': 'json_schema', 'json_schema': {'name': 'name', 'schema': schema_dict, 'strict': True}}`
- Guarantees output matches schema exactly
- Supported fields — object, array, string, number, boolean, enum, ref

### 5.4 When to Use Structured Outputs
- Extracting data from text — entities, attributes
- Classification with fixed labels
- Data pipeline inputs — ensure downstream compatibility
- API responses — type-safe outputs

---

## 6. Function / Tool Calling

### 6.1 Defining Tools
```python
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get current weather for a city',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {'type': 'string', 'description': 'City name'},
                    'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}
                },
                'required': ['city']
            }
        }
    }
]
```

### 6.2 Calling with Tools
```python
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': 'What is the weather in Paris?'}],
    tools=tools,
    tool_choice='auto'  # 'none', 'auto', or {'type':'function','function':{'name':'fn'}}
)
```

### 6.3 Handling Tool Calls
- `response.choices[0].finish_reason == 'tool_calls'` — model wants to call a tool
- `tool_calls = response.choices[0].message.tool_calls`
- `tool_calls[0].function.name` — function name
- `json.loads(tool_calls[0].function.arguments)` — arguments dict
- Execute the function → get result
- Append tool result to messages → continue conversation

### 6.4 Full Tool Call Loop
```python
messages = [{'role': 'user', 'content': 'What is the weather in Paris?'}]

while True:
    response = client.chat.completions.create(model='gpt-4o-mini', messages=messages, tools=tools)
    msg = response.choices[0].message
    messages.append(msg)

    if response.choices[0].finish_reason != 'tool_calls':
        break

    for tool_call in msg.tool_calls:
        result = call_function(tool_call.function.name, json.loads(tool_call.function.arguments))
        messages.append({'role': 'tool', 'tool_call_id': tool_call.id, 'content': str(result)})
```

### 6.5 tool_choice Options
- `'auto'` — model decides whether to call a tool
- `'none'` — model never calls tools
- `'required'` — model must call at least one tool
- `{'type': 'function', 'function': {'name': 'fn'}}` — force specific function

---

## 7. Embeddings

### 7.1 Creating Embeddings
```python
response = client.embeddings.create(
    model='text-embedding-3-small',
    input='Machine learning is a subset of artificial intelligence'
)
embedding = response.data[0].embedding  # list of floats
print(len(embedding))  # 1536 for small, 3072 for large
```

### 7.2 Batch Embeddings
```python
texts = ['text one', 'text two', 'text three']
response = client.embeddings.create(model='text-embedding-3-small', input=texts)
embeddings = [d.embedding for d in response.data]
```

### 7.3 Embedding Models
- `text-embedding-3-small` — 1536 dimensions, cheap, fast
- `text-embedding-3-large` — 3072 dimensions, higher quality
- `text-embedding-ada-002` — legacy, 1536 dimensions

### 7.4 Embedding Dimensions Reduction
- `dimensions=512` parameter — reduce dimensions (3-small and 3-large only)
- Reduces cost and storage without major quality loss
- `client.embeddings.create(model='text-embedding-3-small', input=text, dimensions=256)`

### 7.5 Similarity Computation
- Cosine similarity — `np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))`
- Dot product (normalized) — equivalent to cosine similarity
- `from sklearn.metrics.pairwise import cosine_similarity`

---

## 8. Image Generation (DALL·E)

### 8.1 Generating Images
```python
response = client.images.generate(
    model='dall-e-3',
    prompt='A futuristic data center with glowing blue servers',
    size='1024x1024',
    quality='standard',
    n=1
)
image_url = response.data[0].url
```

### 8.2 Parameters
- `model` — `'dall-e-3'` or `'dall-e-2'`
- `prompt` — text description
- `size` — `'256x256'`, `'512x512'`, `'1024x1024'`, `'1792x1024'`, `'1024x1792'`
- `quality` — `'standard'` or `'hd'` (DALL·E 3 only)
- `style` — `'vivid'` or `'natural'` (DALL·E 3 only)
- `n` — number of images (DALL·E 3: max 1)
- `response_format` — `'url'` (default) or `'b64_json'`

### 8.3 Image Editing and Variations (DALL·E 2)
- `client.images.edit(image=open('img.png','rb'), mask=mask_file, prompt='...', n=1)` — inpainting
- `client.images.create_variation(image=open('img.png','rb'), n=2)` — variations

---

## 9. Audio — Whisper (Transcription) and TTS

### 9.1 Audio Transcription (Whisper)
```python
with open('audio.mp3', 'rb') as f:
    transcript = client.audio.transcriptions.create(
        model='whisper-1',
        file=f,
        language='en',
        response_format='text'
    )
print(transcript)
```

### 9.2 Transcription Parameters
- `model` — `'whisper-1'`
- `file` — audio file object
- `language` — ISO-639-1 code (`'en'`, `'fr'`, `'es'`)
- `response_format` — `'text'`, `'json'`, `'srt'`, `'vtt'`, `'verbose_json'`
- `prompt` — context hint to improve accuracy
- `temperature` — 0 for deterministic

### 9.3 Audio Translation
- `client.audio.translations.create(model='whisper-1', file=f)` — translate to English
- Supports 57 languages

### 9.4 Text-to-Speech (TTS)
```python
response = client.audio.speech.create(
    model='tts-1',
    voice='alloy',
    input='Hello, this is a test of text to speech.'
)
response.stream_to_file('output.mp3')
```

### 9.5 TTS Parameters
- `model` — `'tts-1'` (fast) or `'tts-1-hd'` (higher quality)
- `voice` — `'alloy'`, `'echo'`, `'fable'`, `'onyx'`, `'nova'`, `'shimmer'`
- `response_format` — `'mp3'`, `'opus'`, `'aac'`, `'flac'`, `'wav'`, `'pcm'`
- `speed` — 0.25–4.0 (default: 1.0)

---

## 10. Vision (Multimodal)

### 10.1 Image Input to GPT-4o
```python
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'What is in this image?'},
            {'type': 'image_url', 'image_url': {'url': 'https://example.com/image.jpg'}}
        ]
    }]
)
```

### 10.2 Base64 Image Input
```python
import base64
with open('image.jpg', 'rb') as f:
    b64_image = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'Describe this image'},
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64_image}'}}
        ]
    }]
)
```

### 10.3 Detail Level
- `'detail': 'low'` — faster, fewer tokens, less detail
- `'detail': 'high'` — full resolution analysis, more tokens
- `'detail': 'auto'` — model decides

### 10.4 Multiple Images
- Pass multiple image content items in the same message
- Useful for comparison tasks

---

## 11. Fine-tuning

### 11.1 When to Fine-tune
- Consistent output format not achievable with prompting
- Need model to mimic specific writing style
- Reduce prompt length — embed instructions in weights
- Improve on domain-specific tasks with many examples
- Reduce cost for repetitive tasks (smaller model + fine-tuned)

### 11.2 Fine-tuning Workflow
1. Prepare JSONL training data
2. Upload file — `client.files.create(file=f, purpose='fine-tune')`
3. Create job — `client.fine_tuning.jobs.create(training_file=file_id, model='gpt-4o-mini')`
4. Monitor — `client.fine_tuning.jobs.retrieve(job_id)`
5. Use fine-tuned model — `model='ft:gpt-4o-mini:org:name:id'`

### 11.3 Training Data Format
```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### 11.4 Fine-tuning Parameters
- `model` — base model to fine-tune
- `training_file` — uploaded file ID
- `validation_file` — optional validation set
- `n_epochs` — number of training epochs (default: auto)
- `batch_size` — default: auto
- `learning_rate_multiplier` — default: auto
- `seed` — reproducibility

### 11.5 Fine-tuning API
- `client.fine_tuning.jobs.list()` — list all jobs
- `client.fine_tuning.jobs.retrieve(job_id)` — job status
- `client.fine_tuning.jobs.cancel(job_id)` — cancel
- `client.fine_tuning.jobs.list_events(job_id)` — training events

---

## 12. Assistants API

### 12.1 What is the Assistants API?
- Stateful API — manages conversation threads
- Built-in tool support — Code Interpreter, File Search, Function Calling
- File attachments — upload documents for retrieval
- Persistent threads — no manual conversation history management

### 12.2 Core Components
- **Assistant** — configured model with instructions and tools
- **Thread** — conversation session
- **Message** — user or assistant turn in a thread
- **Run** — execution of an assistant on a thread
- **Step** — individual action within a run

### 12.3 Basic Assistants Workflow
```python
# Create assistant
assistant = client.beta.assistants.create(
    name='Data Analyst',
    instructions='You are a helpful data analysis assistant.',
    model='gpt-4o-mini',
    tools=[{'type': 'code_interpreter'}]
)

# Create thread
thread = client.beta.threads.create()

# Add message
client.beta.threads.messages.create(
    thread_id=thread.id,
    role='user',
    content='Analyze this dataset and find outliers'
)

# Run assistant
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# Get response
messages = client.beta.threads.messages.list(thread_id=thread.id)
print(messages.data[0].content[0].text.value)
```

### 12.4 Assistants Tools
- `{'type': 'code_interpreter'}` — runs Python in sandbox
- `{'type': 'file_search'}` — RAG over uploaded files
- Function tools — same format as chat completions tools

### 12.5 File Upload for Assistants
- `file = client.files.create(file=open('data.csv','rb'), purpose='assistants')`
- Attach to assistant or thread
- `vector_store = client.beta.vector_stores.create(name='docs')`
- Add files — `client.beta.vector_stores.file_batches.create(vector_store_id=vs.id, file_ids=[file.id])`

---

## 13. Moderation

### 13.1 Content Moderation API
```python
response = client.moderations.create(input='Some text to moderate')
result = response.results[0]
print(result.flagged)        # True if harmful
print(result.categories)    # Per-category flags
print(result.category_scores)  # Per-category scores
```

### 13.2 Moderation Categories
- `hate` — hate speech
- `hate/threatening` — hate with violence
- `harassment` — targeted harassment
- `self-harm` — self-harm content
- `sexual` — sexual content
- `sexual/minors` — CSAM
- `violence` — violent content
- `violence/graphic` — graphic violence

---

## 14. Batch API

### 14.1 What is the Batch API?
- Process large volumes of requests asynchronously
- 50% cost reduction vs real-time API
- Up to 24-hour completion window
- Ideal for — embeddings, classification, data enrichment pipelines

### 14.2 Batch Workflow
```python
# Prepare JSONL batch file
# Each line: {"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {...}}

# Upload batch file
batch_file = client.files.create(file=open('batch.jsonl','rb'), purpose='batch')

# Create batch
batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint='/v1/chat/completions',
    completion_window='24h'
)

# Check status
batch = client.batches.retrieve(batch.id)
print(batch.status)  # 'validating', 'in_progress', 'completed', 'failed'

# Download results
if batch.status == 'completed':
    content = client.files.content(batch.output_file_id)
```

---

## 15. Error Handling and Retries

### 15.1 Common Errors
- `openai.APIConnectionError` — network issues
- `openai.RateLimitError` — 429 too many requests
- `openai.APIStatusError` — 4xx/5xx API errors
- `openai.AuthenticationError` — invalid API key
- `openai.BadRequestError` — invalid request parameters

### 15.2 Automatic Retries
- `client = OpenAI(max_retries=3)` — auto-retry on transient errors
- Exponential backoff — built-in
- `client = OpenAI(timeout=30.0)` — custom timeout

### 15.3 Manual Retry Pattern
```python
import time

for attempt in range(3):
    try:
        response = client.chat.completions.create(...)
        break
    except openai.RateLimitError:
        time.sleep(2 ** attempt)
```

---

## 16. Async API

### 16.1 Async Client
```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def get_response():
    response = await async_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': 'Hello'}]
    )
    return response.choices[0].message.content

asyncio.run(get_response())
```

### 16.2 Concurrent Requests with asyncio
```python
async def process_batch(texts):
    tasks = [get_embedding(t) for t in texts]
    return await asyncio.gather(*tasks)
```

---

## 17. Cost and Rate Limit Management

### 17.1 Rate Limits
- TPM — tokens per minute
- RPM — requests per minute
- Limits vary by tier (tier 1–5) and model
- `x-ratelimit-limit-requests` — response header
- `x-ratelimit-remaining-tokens` — remaining tokens

### 17.2 Cost Optimization Strategies
- Use `gpt-4o-mini` for simple tasks
- Use Batch API for offline processing (50% cheaper)
- Cache frequent responses — `functools.lru_cache` or Redis
- Reduce prompt size — shorter system prompts
- Use `max_tokens` to cap output
- Use `text-embedding-3-small` over `ada-002`

### 17.3 Usage Monitoring
- `response.usage.total_tokens` — per request
- Dashboard — platform.openai.com/usage
- Set usage limits — platform.openai.com/account/limits

---

## 18. Security and Best Practices

### 18.1 API Key Security
- Never hardcode API keys in code
- Use environment variables — `os.environ['OPENAI_API_KEY']`
- Use `.env` files with `python-dotenv` — add `.env` to `.gitignore`
- Rotate keys regularly
- Use project-scoped keys (not org-level)
- Set spending limits in dashboard

### 18.2 Prompt Injection Prevention
- Validate and sanitize user input before including in prompts
- Use system message to constrain behavior
- Separate user content from instructions with clear delimiters
- Never trust user-provided text as instructions

### 18.3 Production Best Practices
- Set `request_id` header for tracing — `extra_headers={'X-Request-Id': uuid}`
- Log `response.id` for debugging
- Monitor token usage per request
- Implement rate limit handling with exponential backoff
- Use async client for concurrent workloads

---

## 19. OpenAI Models Comparison

### 19.1 GPT-4o vs GPT-4o mini
- `gpt-4o` — best quality, multimodal, 128k context, more expensive
- `gpt-4o-mini` — fast, cheap, great for most tasks, same context window
- Rule of thumb — default to mini, upgrade to 4o when quality insufficient

### 19.2 o1 / o3 Reasoning Models
- Slower — spends time "thinking" before answering
- Better for — math, logic, complex multi-step reasoning
- Does not support — `temperature`, `top_p`, `system` messages (use `developer` role)
- More expensive — charged for reasoning tokens

### 19.3 Embedding Models
- `text-embedding-3-small` — best cost/quality ratio
- `text-embedding-3-large` — best quality, more dimensions
- `text-embedding-ada-002` — legacy, avoid for new projects

---

## 20. Interview Topics

### 20.1 Frequently Asked Questions
- What is a token and how is it different from a word?
- What is the difference between `temperature` and `top_p`?
- What is the difference between function calling and RAG?
- When would you fine-tune a model vs use few-shot prompting?
- How do you prevent prompt injection attacks?
- What is the Batch API and when to use it?

### 20.2 Coding Problems
- Build a multi-turn chatbot with conversation history
- Implement tool calling for a weather or calculator function
- Extract structured JSON from unstructured text using Structured Outputs
- Build a simple semantic search system using embeddings
- Implement retry logic with exponential backoff for rate limit handling

---

## 21. Real-World Projects

### 21.1 Beginner Projects
- Multi-turn Q&A chatbot
- Text summarizer with adjustable length
- PDF text extractor + Q&A with GPT-4o

### 21.2 Intermediate Projects
- Document Q&A system with embeddings + cosine similarity
- Structured data extractor from free-text inputs
- Customer support chatbot with tool calling

### 21.3 Advanced Projects
- Full RAG pipeline with vector database + GPT-4o
- Multi-tool agent with retry and error handling
- Batch document classifier using Batch API
