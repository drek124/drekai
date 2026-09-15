# DrekAI

A modern, **async-first** Python wrapper for OpenAI-compatible LLM APIs with built-in **tool/function calling** support.

> **Works with OpenAI, Gemini (via proxy), and any OpenAI-compatible endpoint.**

---

## ✨ Features

- **Async-native** – built on `httpx` / `openai` async client
- **Tool use** – define Python functions as AI-callable tools with automatic sandbox parameter injection
- **Streaming** – stream responses token-by-token with a simple callback
- **Sandboxed parameters** – keep secrets invisible to the LLM while still passing them to tools
- **Multi-turn chats** – conversation history management
- **Image support** – send local files, URLs, or base64 images inline
- **Thinking / reasoning** – support for extended-thinking models with effort control
- **Gemini compatibility** – automatic tool-call shimming for Gemini models behind OpenAI proxies

---

## 📦 Installation

```bash
pip install drekai
```

Or install from source:

```bash
git clone https://github.com/drek124/drekai.git
cd drekai
pip install .
```

---

## 🚀 Quick Start

```python
import asyncio
from drekai import Model, Chatbot

# 1. Define a model endpoint
model = Model("gpt-4o", "https://api.openai.com/v1", api_key="sk-...")

# 2. Create a chatbot
bot = Chatbot(
    name="Helper",
    system_prompt="You are a helpful assistant.",
    parent_model=model,
    temperature=0.7,
)

async def main():
    # One-shot generation
    response = await bot.generate_text("Hello!")
    print(response.choices[0].message.content)

    # Multi-turn chat
    chat = bot.start_chat()
    r1 = await chat.generate_reply("What's the weather in Tokyo?")
    r2 = await chat.generate_reply("And in London?")
    print(r2.choices[0].message.content)

asyncio.run(main())
```

---

## 🛠️ Tool Use

```python
from drekai import Model, Chatbot
from drekai.tools import Tool, ToolParameter

model = Model("gpt-4o", "https://api.openai.com/v1", api_key="sk-...")
bot = Chatbot("Assistant", "You are a helpful assistant.", model)

async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In a real app, call a weather API here
    return f"The weather in {city} is sunny, 25°C."

tools = [
    Tool(
        "get_weather",
        [ToolParameter("city", "City name", type=str)],
        callback=get_weather,
    )
]

async def main():
    chat = bot.start_chat()
    response = await chat.generate_reply(
        "What's the weather in Paris?",
        tools=tools,
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

### Sandboxed Parameters

Keep secrets like user IDs invisible to the LLM:

```python
async def get_friends(user, limit: int = 50) -> str:
    """Fetch the user's friends list."""
    return str(user.get_friends(limit=limit))

tools = [
    Tool(
        "get_friends",
        [ToolParameter("limit", "Max friends to return", required=False, type=int)],
        callback=get_friends,
        sandbox_params=["user"],  # hidden from the LLM
    )
]

await chat.generate_reply(
    "Who are my friends?",
    tools=tools,
    sandbox_params={"user": current_user},  # injected at call time
)
```

---

## 🖼️ Images

```python
from drekai.messaging import Image

# Local file
await chat.generate_reply(
    "Describe this image.",
    items=[Image("/path/to/photo.jpg")],
)

# URL
await chat.generate_reply(
    "Describe this image.",
    items=[Image("https://example.com/photo.jpg")],
)

# Base64
await chat.generate_reply(
    "Describe this image.",
    items=[Image(base64_data, b64=True)],
)
```

---

## 📖 API Reference

### `Model(model_id, base_url, *, api_key)`
Root model representing an API endpoint.

### `Chatbot(name, system_prompt, parent_model, *, api_key, temperature, thinking, reasoning_effort)`
A named chatbot built on a model.

### `Chat`
Created via `chatbot.start_chat()`. Manages conversation history.

| Method | Description |
|---|---|
| `generate_reply(...)` | Generate the next response |
| `add_message_to_context(content, role)` | Manually add a message |
| `clear()` | Reset history (keeps system prompt) |
| `delete_first_message(role)` | Remove first message with given role |
| `stop_live_generation()` | Stop an active stream mid-generation |

### `ChatSettings(max_tokens, show_tool_error_type)`
Per-chat configuration.

### `Tool(name, params, callback, sandbox_params)`
An AI-callable function.

### `ToolParameter(name, description, type, required)`
Describes a tool parameter.

### `MessageItem` / `Image`
Chat message attachments.

---

## 🧪 Development

```bash
# Clone and install in editable mode with dev deps
git clone https://github.com/drek124/drekai.git
cd drekai
pip install -e ".[dev]"

# Lint
ruff check drekai/

# Type check
mypy drekai/

# Test
pytest
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
