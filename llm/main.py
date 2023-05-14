"""Simple LLM API for Python."""

# Usage:
# import llm
# llm.complete("hello, I am an animal called") --> "cat" # Uses GPT-3 by default if key is provided
# llm.complete("hello, I am an animal called", engine="huggingface/roberta-base") --> "cat"
# llm.chat(["hello", "hi", "how are you?"], system="Behave like a goat.")
# Also try:
# llm.setup_cache() and all requests will be cached

import json
import os

from llm.util import parse_args, structure_chat, configure_api_keys
from llm.api import openaiapi, claude


def complete(prompt, engine="text-davinci-003", **kwargs):
    args = parse_args(engine, **kwargs)
    if args.service == "openai":
        return openaiapi.complete(prompt, args.engine, **args.kwargs)
    elif args.service == "anthropic":
        return claude.complete(prompt, args.engine, **args.kwargs)
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
        result = claude.chat(messages, args.engine, **args.kwargs)
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
        result = claude.stream_chat(messages, args.engine, **args.kwargs)
    else:
        raise ValueError(f"Engine {engine} is not supported.")
    async for message in result:
        yield message.strip()


def embed(text, engine="text-davinci-003", **kwargs):
    """Embed text using the LLM API."""
    raise NotImplementedError("Embedding is not yet implemented.")


def set_api_key(**key_thing):
    """Set the OpenAI API key. Call me like this:
    llm.set_api_keys(openai="sk-...", anthropic="sk-...")
    or
    llm.set_api_keys("path/to/api_keys.json")
    """
    return configure_api_keys(**key_thing)
