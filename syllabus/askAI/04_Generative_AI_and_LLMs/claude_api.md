Yes. If you're going to give the AI **only one Anthropic Claude API section at a time**, use this as your reusable **master Anthropic Claude API prompt**. Each time, just replace `[CLAUDE SECTION]`.

I am learning the Anthropic Claude API as part of my AI/ML and Generative AI learning journey.

I will give you **one Claude API roadmap section at a time**. Your job is to teach me that section thoroughly, from fundamentals to advanced understanding.

Assume that I know basic Python and have some exposure to LLMs and the OpenAI API, but I want to develop a **strong practical and conceptual understanding** of the Anthropic Claude API — its unique design philosophy, its differentiating features (Constitutional AI, extended thinking, prompt caching, tool use), and how to use it to build production Generative AI applications.

## Claude API Section

[CLAUDE SECTION]

---

# How I want you to teach me

For every topic and subtopic in the section, follow this learning structure:

### 1. What is it?

- Give a clear definition.
- Explain the problem it solves or the capability it provides.
- Explain how it differs from the equivalent OpenAI API feature (when comparison is meaningful).
- Place it in the Claude API surface:
  - Messages API → Extended Thinking → Tool Use → Prompt Caching → Vision → Files API → Batch API

### 2. Intuition

- Explain the concept in very simple language.
- Use a real-world analogy when useful.
- Explain the key design philosophy behind Anthropic's approach to this feature.
- Then gradually move toward the technical explanation.

### 3. Prerequisites

- Mention any Python/HTTP/LLM/API concepts I need to understand first.
- If necessary, briefly teach the prerequisite before continuing.

### 4. API Design and Core Concepts

- Show the complete Python SDK code using the **modern `anthropic` Python SDK**.
- Always use `import anthropic; client = anthropic.Anthropic()`.
- Explain every parameter with its type, default value, and effect.
- Explain the structure of the API response object.

```python
import anthropic

client = anthropic.Anthropic(api_key="sk-ant-...")  # or set ANTHROPIC_API_KEY env var

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    system="You are a helpful AI assistant.",
    messages=[
        {"role": "user", "content": "What is the difference between RAG and fine-tuning?"}
    ]
)

print(message.content[0].text)
print(message.usage)  # input_tokens, output_tokens
```

- Explain the `messages` list structure: user and assistant roles (no system role in messages — system is a top-level parameter).
- Explain `model`, `max_tokens` (required), `temperature`, `top_p`, `top_k`, `stop_sequences`.
- Explain `message.content` (list of content blocks), `message.usage`, `message.stop_reason`.

### 5. Claude's Unique Design Differences vs. OpenAI

Explain the key API design differences between Claude and OpenAI:

| Feature         | Claude (Anthropic)                               | GPT (OpenAI)                          |
| --------------- | ------------------------------------------------ | ------------------------------------- |
| System prompt   | Top-level `system` param                         | Inside `messages` as `role: system`   |
| `max_tokens`    | **Required**                                     | Optional (has default)                |
| Response format | `message.content[0].text`                        | `response.choices[0].message.content` |
| Streaming       | `client.messages.stream()` context manager       | `stream=True` parameter               |
| Tool use        | `tools` + `tool_use` content blocks              | `tools` + `tool_calls`                |
| Thinking        | `extended_thinking` (unique to Claude)           | `o1`/`o3` reasoning models            |
| Prompt caching  | `cache_control: {"type": "ephemeral"}`           | Automatic prefix caching              |
| Models          | claude-3-5-sonnet, claude-opus-4-5, claude-haiku | gpt-4o, gpt-4o-mini, o1               |

### 6. Examples

Give examples progressively:

**Beginner → Intermediate → Advanced**

For every important example:

- Show the complete, runnable code.
- Explain it line by line.
- Show the expected response structure.
- Explain what Claude is doing and why the response has that structure.
- Show token usage and cost.

Use realistic Generative AI use cases:

- Document analysis and summarization
- Multi-turn coding assistant
- Structured data extraction with tool use
- Long-document processing with prompt caching
- Extended thinking for complex reasoning tasks
- Vision: analyzing charts, diagrams, and screenshots

### 7. Extended Thinking (Claude-Specific)

Explain Claude's Extended Thinking feature in depth — a key differentiator:

```python
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # thinking token budget
    },
    messages=[{
        "role": "user",
        "content": "Solve this step by step: If a train travels 120km in 1.5 hours, what is its average speed? Now use this to calculate how far it travels in 2h 20min."
    }]
)

# Response has two content blocks: thinking + text
for block in response.content:
    if block.type == "thinking":
        print("THINKING:", block.thinking)
    elif block.type == "text":
        print("ANSWER:", block.text)
```

- Explain what Extended Thinking is: Claude reasons through a scratchpad before answering
- When to use it: complex math, multi-step reasoning, hard coding problems
- The `budget_tokens` parameter: thinking token budget (separate from output tokens)
- Cost: thinking tokens are billed as output tokens
- Comparison to OpenAI's `o1`/`o3` models

### 8. Tool Use (Function Calling)

Explain Claude's tool use system in depth:

```python
tools = [{
    "name": "search_documents",
    "description": "Search through internal documents for relevant information.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 5
            }
        },
        "required": ["query"]
    }
}]

# Step 1: Claude decides to use a tool
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What does our policy say about remote work?"}]
)

# Step 2: Execute the tool
if response.stop_reason == "tool_use":
    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    tool_result = search_documents(**tool_use_block.input)  # your function

    # Step 3: Return result to Claude
    final_response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "What does our policy say about remote work?"},
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": tool_result
            }]}
        ]
    )
```

- Explain `tool_use` content blocks vs. `tool_result` messages
- Explain `stop_reason: "tool_use"` as the signal that Claude wants to call a tool
- Explain parallel tool use: Claude can call multiple tools in one response
- Compare to OpenAI's `tool_calls` structure

### 9. Prompt Caching (Claude-Specific)

Explain Anthropic's Prompt Caching — a major cost-saving feature for RAG and repeated context:

```python
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a helpful assistant with access to our company knowledge base.",
        },
        {
            "type": "text",
            "text": "<documents>" + large_document_context + "</documents>",
            "cache_control": {"type": "ephemeral"}  # Mark for caching
        }
    ],
    messages=[{"role": "user", "content": "What is our refund policy?"}]
)

print(response.usage.cache_creation_input_tokens)  # first call: tokens written to cache
print(response.usage.cache_read_input_tokens)       # subsequent calls: tokens read from cache
```

- How prompt caching works: Anthropic caches the marked portion for 5 minutes
- Cost: cache creation = 1.25× normal, cache read = 0.1× normal (90% discount)
- When to use: large system prompts, long document contexts in RAG, repeated tool definitions
- Minimum cacheable block size: 1024 tokens (Sonnet/Haiku), 2048 tokens (Opus)
- Comparison to OpenAI's automatic prefix caching

### 10. Vision and Multi-Modal

Explain Claude's vision capabilities:

```python
import base64

with open("chart.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data
                }
            },
            {
                "type": "text",
                "text": "Analyze this chart and extract the key data points as JSON."
            }
        ]
    }]
)
```

- Supported formats: JPEG, PNG, GIF, WebP
- URL-based images: `"type": "url", "url": "https://..."`
- Use cases: document parsing, chart analysis, screenshot Q&A, diagram understanding
- Token cost: images are billed by pixel count

### 11. Streaming

Explain Claude's streaming API:

```python
# Context manager pattern (recommended)
with client.messages.stream(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a short story."}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    final_message = stream.get_final_message()
    print(f"\nTotal tokens: {final_message.usage.output_tokens}")
```

- Explain the `stream()` context manager vs. `stream=True` parameter approach
- How to get the final message with complete usage stats
- Streaming with tool use: handling `input_json_delta` events
- FastAPI integration: `StreamingResponse` with async streaming

### 12. XML Prompting — Claude Best Practice

Explain Claude's preference for XML-structured prompts:

```python
system = """You are a data extraction assistant.

When extracting information, always structure your response using XML tags:
<extraction>
  <field name="company">...</field>
  <field name="date">...</field>
  <field name="amount">...</field>
</extraction>"""

# In Claude prompts, use XML tags for:
# - Separating document content: <document>...</document>
# - Labeling instructions: <instructions>...</instructions>
# - Marking examples: <example>...</example>
# - Structuring output: <analysis>...</analysis>
```

- Why Claude responds well to XML structure
- XML injection defense: sanitizing user input in XML-tagged prompts
- When to use XML vs. JSON response format
- How XML tags help Claude organize long documents

### 13. Cost Management

Explain Claude pricing and optimization:

| Model             | Input (per 1M tokens) | Output (per 1M tokens) | Best for                             |
| ----------------- | --------------------- | ---------------------- | ------------------------------------ |
| claude-haiku-3-5  | $0.80                 | $4.00                  | Simple tasks, high volume            |
| claude-sonnet-4-5 | $3.00                 | $15.00                 | Most production tasks                |
| claude-opus-4-5   | $15.00                | $75.00                 | Complex reasoning, extended thinking |

**Optimization strategies:**

- Use `claude-haiku-3-5` for classification, routing, simple extraction
- Use prompt caching for RAG contexts and long system prompts (90% read discount)
- Use the Batch API for offline processing (50% discount, 24h SLA)
- Count tokens with `client.messages.count_tokens()` before sending
- Use `max_tokens` wisely — you are billed for output tokens actually generated

### 14. Error Handling

Explain how to handle Claude API errors in production:

```python
import anthropic
from anthropic import RateLimitError, APIConnectionError, APIStatusError
import time

client = anthropic.Anthropic()

def call_with_retry(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.messages.create(
                model="claude-opus-4-5",
                max_tokens=1024,
                messages=messages
            )
        except RateLimitError:
            time.sleep(2 ** attempt)
        except APIConnectionError as e:
            raise RuntimeError(f"Connection error: {e}")
        except APIStatusError as e:
            raise RuntimeError(f"API error {e.status_code}: {e.message}")
    raise RuntimeError("Max retries exceeded")
```

### 15. Async Usage

```python
import anthropic
import asyncio

client = anthropic.AsyncAnthropic()

async def get_answer(question: str) -> str:
    message = await client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": question}]
    )
    return message.content[0].text
```

### 16. Common Mistakes

Show common mistakes beginners make.

For each mistake:

**Wrong approach → Why it is wrong → Correct approach**

Also mention important pitfalls such as:

- Putting the system prompt inside `messages` as `{"role": "system"}` — Claude uses a top-level `system` parameter
- Forgetting `max_tokens` — it is **required** in the Claude API (no default)
- Accessing `response.choices[0]` (OpenAI pattern) — Claude uses `message.content[0].text`
- Not checking `stop_reason == "tool_use"` before assuming a text response
- Marking too-short content with `cache_control` — below minimum tokens, caching is ignored
- Not sanitizing user input inside XML-tagged prompts — XML injection risk

### 17. Best Practices

Give practical rules I should follow when using the Claude API in production.

Include:

- Always store API keys in environment variables — never in code
- Always include `max_tokens` — it is required
- Use `claude-haiku-3-5` as the default; upgrade only when quality is insufficient
- Use XML tags for structured prompts and complex document contexts
- Use prompt caching for any RAG system with large repeated context
- Use `AsyncAnthropic` in FastAPI routes
- Monitor `stop_reason` in every response — `"max_tokens"` means the output was cut off
- Any section-specific best practices

### 18. Interview Perspective

Give me important interview questions related to the section.

Include:

- Conceptual questions (e.g., "What is Constitutional AI and how does it affect Claude's responses?")
- API design questions (e.g., "How does Claude's tool use differ from OpenAI's function calling?")
- Feature questions (e.g., "What is Extended Thinking and when would you use it?")
- Cost questions (e.g., "How would you use prompt caching to reduce costs in a RAG application?")
- Debugging questions (e.g., "Your Claude API call returns `stop_reason: max_tokens` — what do you do?")

Provide the answers and explanations after the questions.

### 19. Practice

Give me exercises at three levels:

**Beginner** — 2–3 exercises (basic messages call, streaming, token counting)

**Intermediate** — 2–3 exercises (tool use round-trip, prompt caching, XML-structured extraction)

**Advanced** — 2–3 exercises (extended thinking for multi-step problem, vision + extraction, async RAG pipeline)

Do NOT immediately provide solutions. Let me solve them first.

### 20. Mini Project

Give me a small Claude API mini-project using the concepts in this section.

Give me:

- Problem statement (e.g., build a document analysis pipeline with prompt caching and structured XML extraction)
- Required Claude API features
- Expected input/output
- Cost estimate with and without caching
- Constraints
- Hints

Do not provide the complete solution until I attempt it.

---

# Important Teaching Rules

1. Always use the **modern `anthropic` Python SDK** — no raw HTTP calls.
2. Always highlight **differences from the OpenAI API** — many learners come from OpenAI.
3. Always show **token usage and cost** for significant examples.
4. Always show **error handling** in production code examples.
5. When tool use is shown, always show the complete **3-step round-trip** (call → execute → result).
6. When prompt caching is shown, always show `cache_creation_input_tokens` vs. `cache_read_input_tokens` in the response.
7. When Extended Thinking is shown, always show how to access the `thinking` content block separately from the `text` block.
8. Do not introduce unrelated Claude API topics.
9. Teach ONLY the section I provide.
10. At the end, give a concise **Revision Sheet**: key SDK classes and methods, Claude vs. OpenAI quick comparison, model tier guide, prompt caching guide, error handling reference, common mistakes, best practices.
11. After teaching the section, STOP and wait for me.

---

# Teaching Goal

My goal is not simply to call `client.messages.create()`.

I want to understand:

**What → Why → API Design → Claude's Unique Features → Tool Use → Extended Thinking → Prompt Caching → Vision → Production Reliability**

so that I can confidently build, optimize, and scale Generative AI applications using the Anthropic Claude API, and make informed decisions about when to use Claude vs. OpenAI.

Now teach me the following Claude API section:

[CLAUDE SECTION]

**How to use it:** paste the prompt once, then replace `[CLAUDE SECTION]` with something like `3. Tool Use and Agentic Patterns` or `3.1 Multi-Step Tool Calling with Parallel Tool Use`. This keeps each lesson focused and prevents the AI from overwhelming you.
