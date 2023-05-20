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
import asyncio

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
    initial_response_done = False
    async for raw_token in result:
        token, output_cache = format_streaming_output(
            raw_token, stream_method, args.service, output_cache)
        if args.service == "anthropic":
            # First token of Anthropic output always starts with an empty space.
            # If we're in delta mode, strip the space only if this is the first token out.
            # If we're in full mode, strip the space every time.
            if stream_method == "delta" and not initial_response_done:
                token = token.lstrip()
            elif stream_method == "default" or stream_method == "full":
                token = token.lstrip()
            initial_response_done = True
        yield token


async def multi_stream_chat(messages, engines=["anthropic:claude-instant-v1", "openai:gpt-3.5-turbo"], **kwargs):
    """Chat with multiple LLM APIs simultaneously.

    engines should be a list of engine strings, e.g. 
    ["openai:gpt-3.5-turbo", "anthropic:claude-v1"]

    The responses will be yielded in the order they are received from the engines.
    Each response is returned as a tuple of the form (engine_name, response).
    """
    # Create a list of streams, each one labeled with the name of the engine
    streams = [(engine, stream_chat(messages, engine, "full", **kwargs))
               for engine in engines]

    while streams:  # While there are still streams left
        for i, (engine, stream) in enumerate(streams):
            try:
                result = await stream.__anext__()
                # Yield the result along with the engine name
                yield (engine, result)
            except StopAsyncIteration:  # The stream has ended
                del streams[i]  # Remove it from the list


def set_api_key(*args, **kwargs):
    """Set the OpenAI API key. Call me like this:
    - llm.set_api_keys(openai="sk-...")
    - llm.set_api_keys("path/to/api_keys.json")
    """
    return configure_api_keys(*args, **kwargs)
