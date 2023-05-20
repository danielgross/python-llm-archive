# python-llm: A LLM API for Humans

![Tests](https://github.com/danielgross/python-llm/actions/workflows/tests.yml/badge.svg)

The vision is to be as simple as python-requests, but for LLMs. This library supports models from OpenAI and Anthropic, with more coming soon.

## Usage

```python
import llm
llm.set_api_key(openai="sk-...", anthropic="sk-...")

# Chat
llm.chat(["what is 2+2"]) # 4. Uses GPT-3 by default if key is provided.
llm.chat(["what is 2+2"], engine="anthropic:claude-instant-v1") # 4.
llm.stream_chat(["what is 2+2"]) # 4. 
llm.multi_stream_chat(["what is 2+2"], engines=["anthropic:claude-instant-v1", 
                      "openai:gpt-3.5-turbo"])

# Completion
llm.complete("hello, I am") # A GPT model.
llm.complete("hello, I am", engine="openai:gpt-4") # A big GPT model.
llm.complete("hello, I am ", engine="anthropic:claude-instant-v1") # Claude.

# Back-and-forth chat [human, assistant, human]
llm.chat(["hi", "hi there, how are you?", "good, tell me a joke"]) # Why did chicken cross road?

# Engines are in the provider:model format, as in openai:gpt-4, or anthropic:claude-instant-v1.
```

## Installation

To install `python-llm`, use pip: ```pip install python-llm```.

## Configuration
You can set API keys in a few ways:
1. Through environment variables (you can also set a `.env` file).
```bash
export OPENAI_API_KEY=sk_...
export ANTHROPIC_API_KEY=sk_...
```
2. By calling the method manually:
```python
import llm
llm.set_api_key(openai="sk-...", anthropic="sk-...")
```
3. By passing a JSON file like this:
```python
llm.set_api_key("path/to/api_keys.json")
```
The JSON should look like:
```json
{
  "openai": "sk-...",
  "anthropic": "sk-..."
}
```
