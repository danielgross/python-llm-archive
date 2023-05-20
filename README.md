# python-llm: A LLM API for Humans

![Tests](https://github.com/danielgross/python-llm/actions/workflows/tests.yml/badge.svg)

The simplicity and elegance of python-requests, but for LLMs. This library supports models from OpenAI and Anthropic. I will try to add more when I have the time, and am warmly accepting pull requests if that's of interest.

## Usage

```python
import llm
llm.set_api_key(openai="sk-...", anthropic="sk-...")

# Chat
llm.chat("what is 2+2") # 4. Uses GPT-3 by default if key is provided.
llm.chat("what is 2+2", engine="anthropic:claude-instant-v1") # 4.

# Completion
llm.complete("hello, I am") # A GPT model.
llm.complete("hello, I am", engine="openai:gpt-4") # A big GPT model.
llm.complete("hello, I am ", engine="anthropic:claude-instant-v1") # Claude.

# Back-and-forth chat [human, assistant, human]
llm.chat(["hi", "hi there, how are you?", "good, tell me a joke"]) # Why did chicken cross road?

# Streaming chat
llm.stream_chat(["what is 2+2"]) # 4. 
llm.multi_stream_chat(["what is 2+2"], 
                      engines=
                      ["anthropic:claude-instant-v1", 
                      "openai:gpt-3.5-turbo"]) 
# Results will stream back to you from both models at the same time like this:
# ["anthropic:claude-instant-v1", "hi"], ["openai:gpt-3.5-turbo", "howdy"], 
# ["anthropic:claude-instant-v1", " there"] ["openai:gpt-3.5-turbo", " my friend"]

# Engines are in the provider:model format, as in openai:gpt-4, or anthropic:claude-instant-v1.
```

## Multi Stream Chat In Action
Given this feature is very lively, I've included a video of it in action.

https://github.com/danielgross/python-llm/assets/279531/d68eb843-7a32-4ffe-8ac2-b06b81e764b0

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

## TODO 
- [ ] Caching! 
- [ ] More LLM vendors!
- [ ] More tests!

