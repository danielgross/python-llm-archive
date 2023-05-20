"""Simple LLM API for Python."""

# Usage:
# import llm
# llm.complete("hello, I am an animal called") --> "cat" # Uses GPT-3 by default if key is provided
# llm.complete("hello, I am an animal called", engine="huggingface/roberta-base") --> "cat"
# llm.chat(["hello", "hi", "how are you?"], system="Behave like a goat.")
# Also try:
# llm.setup_cache() and all requests will be cached

from llm.utils.parsing import parse_args, structure_chat, format_streaming_output
from llm.utils.apikeys import load_keys_from_cache, configure_api_keys
from llm.api import anthropicapi, openaiapi


# Try loading keys from cache
load_keys_from_cache()


def complete(prompt, engine="openai:text-davinci-003", **kwargs):
    args = parse_args(engine, **kwargs)
    if args.service == "openai":
        result = openaiapi.complete(prompt, args.engine, **args.kwargs)
    elif args.service == "anthropic":
        result = anthropicapi.complete(prompt, args.engine, **args.kwargs)
    else:
        raise ValueError(f"Engine {engine} is not supported.")
    return result.strip()


# Can also pass in system="Behave like a bunny rabbit" for system message.
def chat(messages, engine="openai:gpt-3.5-turbo", **kwargs):
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


async def stream_chat(messages, engine="openai:gpt-3.5-turbo", stream_method="default", **kwargs):
    """Chat with the LLM API.
    stream_method="default" uses the default streaming method for the engine. 
    For OpenAI this will be "delta" and for Anthropic (which restreams the 
    output on every `yield`, it would be "full".
    """
    args = parse_args(engine, **kwargs)
    messages = structure_chat(messages)
    if args.service == "openai":
        result = openaiapi.stream_chat(messages, args.engine, **args.kwargs)
    elif args.service == "anthropic":
        result = anthropicapi.stream_chat(messages, args.engine, **args.kwargs)
    else:
        raise ValueError(f"Engine {engine} is not supported.")

    output_cache = []
    async for raw_token in result:
        token, output_cache = format_streaming_output(
            raw_token, stream_method, args.service, output_cache)
        if stream_method != "delta":
            token = token.strip()
        yield token


def set_api_key(*args, **kwargs):
    """Set the OpenAI API key. Call me like this:
    - llm.set_api_keys(openai="sk-...")
    - llm.set_api_keys("path/to/api_keys.json")
    """
    return configure_api_keys(*args, **kwargs)
