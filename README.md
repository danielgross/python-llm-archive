# python-llm: A LLM API for Humans

The vision is to be as simple as python-requests, but for LLMs. It supports models from OpenAI and Anthropic, with more coming soon.

![Tests](https://github.com/danielgross/python-llm/actions/workflows/tests.yml/badge.svg)

## Usage

```python
import llm
llm.set_api_key(openai="sk-...", anthropic="sk-...")

# Chat
llm.chat(["what is 2+2"]) # 4. Uses GPT-3 by default if key is provided.
llm.chat(["what is 2+2"], engine="anthropic:claude-instant-v1") # 4.

# Completion
llm.complete("hello, I am") # A GPT model.
llm.complete("hello, I am", engine="openai:gpt-4") # A big GPT model.
llm.complete("hello, I am ", engine="anthropic:claude-instant-v1") # Claude.

# Back-and-forth chat
llm.chat(["hi", "hi there, how are you?", "good, tell me a joke"]) # Human/assistant/human exchanges are supported.

# Embedding
llm.embed(open("harrypotter.txt").read()).tsne() # (I haven't implemented this yet.)

# Engines are in the provider:model format, as in openai:gpt-4, or anthropic:claude-instant-v1.
```

## Installation

To install `python-llm`, use pip: ```pip install python-llm```.