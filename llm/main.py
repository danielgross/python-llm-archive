"""Simple LLM API for Python."""

# Usage:
# import llm
# llm.complete("hello, I am an animal called") --> "cat" # Uses GPT-3 by default if key is provided
# llm.complete("hello, I am an animal called", engine="huggingface/roberta-base") --> "cat"
# llm.chat(["hello", "hi", "how are you?"], system="Behave like a goat.")
# Also try:
# llm.setup_cache() and all requests will be cached

from llm.utils.parsing import parse_args, structure_chat
from llm.utils.apikeys import load_keys_from_cache, configure_api_keys
from llm.api import anthropicapi, openaiapi


# Try loading keys from cache
load_keys_from_cache()


def complete(prompt, engine="openai:text-davinci-003", **kwargs):
    args = parse_args(engine, **kwargs)
    if args.service == "openai":
        return openaiapi.complete(prompt, args.engine, **args.kwargs)
    elif args.service == "anthropic":
        return anthropicapi.complete(prompt, args.engine, **args.kwargs)
    else:
        raise ValueError(f"Engine {engine} is not supported.")


# Can also pass in system="Behave like a bunny rabbit" for system message.
def chat(messages, engine="openai:text-davinci-003", **kwargs):
    """Chat with the LLM API."""
    args = parse_args(engine, **kwargs)
    messages = structure_chat(messages)
    if args.service == "openai":
        result = openaiapi.chat(messages, args.engine, **args.kwargs)
    elif args.service == "anthropic":
        result = anthropicapi.chat(messages, args.engine, **args.kwargs)
    else:
        raise ValueError(f"Engine {engine} is not supported.")
    return result.strip()


async def stream_chat(messages, engine="text-davinci-003", **kwargs):
    """Chat with the LLM API."""
    args = parse_args(engine, **kwargs)
    messages = structure_chat(messages)
    if args.service == "openai":
        NotImplementedError("Not implemented yet.")
    elif args.service == "anthropic":
        result = anthropicapi.stream_chat(messages, args.engine, **args.kwargs)
    else:
        raise ValueError(f"Engine {engine} is not supported.")
    async for message in result:
        yield message.strip()


def embed(text, engine="text-davinci-003", **kwargs):
    """Embed text using the LLM API."""
    raise NotImplementedError("Embedding is not yet implemented.")


def set_api_key(*args, **kwargs):
    """Set the OpenAI API key. Call me like this:
    - llm.set_api_keys(openai="sk-...")
    - llm.set_api_keys("path/to/api_keys.json")
    """
    return configure_api_keys(*args, **kwargs)
