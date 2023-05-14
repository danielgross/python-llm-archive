# python-llm: A LLM API for Humans

This is a Python wrapper for the Language Learning Models (LLM) APIs, designed to make it easy to use different language models. It supports models from OpenAI and Anthropic, with a simple and intuitive interface.

![Tests](https://github.com/danielgross/python-llm/actions/workflows/tests.yml/badge.svg)

## Features

- **Text Completion**: Complete text prompts using different language models.
- **Chat**: Simulate a conversation with a language model.
- **Text Embedding**: Transform text into a high-dimensional vector (not yet implemented).

## Installation

To install `python-llm`, use pip: ```pip install python-llm```.

## Usage

```python
import llm

# Completion
llm.complete("hello, I am a") # Uses GPT-3 by default if key is provided.
llm.complete("hello, I am ", engine="anthropic:claude-instant-v1") # Uses Anthropic's model.

# Chat
llm.chat(["hi", "hi how are you", "tell me a joke"])

# Embedding 
llm.embed(open("harrypotter.txt").read())

# Engines are in the provider:model format, as in openai:gpt-4, or anthropic:claude-instant-v1.
```

## Configuration
You can configure the API keys for the various services using the set_api_key method:
```python
llm.set_api_key(openai="sk-...", natdev="...")
# or
llm.set_api_key("path/to/api_keys.json")
```
