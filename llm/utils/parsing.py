import collections


def parse_args(engine, **kwargs):
    """Parse args."""
    assert ":" in engine, "Engine must be in the format 'engine:args', as in openai:text-davinci-003"
    service, engine = engine.split(':', 1)
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
