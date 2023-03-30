# Return a dot dictionary
import collections
import logging
import json
import os


def parse_args(engine, **kwargs):
    """Parse args."""
    assert ":" in engine, "Engine must be in the format 'engine:args', as in openai:text-davinci-003"
    service, engine = engine.split(':', 1)
    return collections.namedtuple("Args", ["service", "engine", "kwargs"])(service, engine, kwargs)


api_keys = {}


def configure_api_keys(**key_thing):
    """Set the OpenAI API key. Call me like this:
    llm.set_api_keys(openai="sk-...")
    or
    llm.set_api_keys("path/to/api_keys.json")
    """
    # If it's a file, load it
    if isinstance(key_thing, str) and os.path.exists(key_thing):
        with open(key_thing) as f:
            api_keys.update(json.load(f))
    elif not isinstance(key_thing, dict):
        raise ValueError(
            "API key must be a dictionary or a path to a JSON file.")
    else:
        api_keys.update(key_thing)
    _on_update_api_keys()


def _on_update_api_keys():
    """Update the API keys from the environment."""
    import openai
    logging.debug(f"Setting OpenAI API key to {api_keys.get('openai')}")
    openai.api_key = api_keys.get("openai")


def structure_chat(messages):
    """Structure chat messages."""
    if isinstance(messages[0], str):
        # recreate as dictionary where "role" is either "system" or "user"
        messages = [{"role": "user" if i % 2 == 0 else "assistant",
                     "content": text} for i, text in enumerate(messages)]
    elif isinstance(messages[0], str):
        raise ValueError("Chat messages must be strings or dictionaries.")
    else:
        messages = [{"role": message.get(
            "role", "user"), "content": message["content"]} for message in messages]
    return messages
