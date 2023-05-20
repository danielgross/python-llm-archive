import collections


def parse_args(engine, **kwargs):
    """Parse args."""
    assert ":" in engine, "Engine must be in the format 'engine:args', as in openai:text-davinci-003"
    service, engine = engine.split(':', 1)
    if service == "anthropic":
        kwargs["max_tokens_to_sample"] = kwargs.pop("max_tokens", 100)
    return collections.namedtuple("Args", ["service", "engine", "kwargs"])(service, engine, kwargs)


def structure_chat(messages):
    """Structure chat messages."""
    if isinstance(messages[0], str):
        # recreate as dictionary where "role" is either "assistant" or "user"
        messages = [{"role": "user" if i % 2 == 0 else "assistant",
                     "content": text} for i, text in enumerate(messages)]
    elif isinstance(messages[0], str):
        raise ValueError("Chat messages must be strings or dictionaries.")
    else:
        messages = [{"role": message.get(
            "role", "user"), "content": message["content"]} for message in messages]
    return messages


def format_streaming_output(raw_tokens, stream_method, service, output_cache):
    if stream_method == "default":
        output_cache.append(raw_tokens)
        return raw_tokens, output_cache
    elif stream_method == "full" and service == "anthropic":
        output_cache.append(raw_tokens)
        return raw_tokens, output_cache
    elif stream_method == "delta" and service == "openai":
        output_cache.append(raw_tokens)
        return raw_tokens, output_cache
    elif stream_method == "delta" and service == "anthropic":
        # "raw_tokens" contain the string repeated from start, and we
        # want to yield only the new part.
        # output_cache is a list of raw_tokens
        streaming_output = raw_tokens[len(''.join(output_cache)):]
        output_cache = [raw_tokens]
        return streaming_output, output_cache
    elif stream_method == "full" and service == "openai":
        # Here the situation is reversed: raw_tokens contain only the
        # new part, and we want to yield the full string.
        streaming_output = ''.join(output_cache) + raw_tokens
        output_cache.append(raw_tokens)
        return streaming_output, output_cache
    else:
        raise ValueError(
            f"Stream method {stream_method} is not supported for service {service}.")
