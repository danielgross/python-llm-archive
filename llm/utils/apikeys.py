import logging
import json
import os

from dotenv import load_dotenv

# Useful os variables
py_llm_dir = os.path.join(os.path.expanduser('~'), ".python-llm")
env_path = os.path.join(py_llm_dir, "apikeys.env")

# Supported apis and memory
supported_apis = ["openai", "anthropic"]
api_keys = {}


# --- public interface ---

def configure_api_keys(*args, **kwargs) -> None:
    """Configure the API keys. Call me like this:
    - llm.set_api_keys(openai="sk-...")
    - llm.set_api_keys("path/to/api_keys.json")
    """

    if args and len(args) > 1 and not isinstance(args[0], str):
        raise ValueError(
            "Only one positional argument (str) is allowed for filepath.")

    # load api keys from a json filepath
    elif args and len(args) == 1 and isinstance(args[0], str):
        filepath = args[0]
        if not os.path.exists(filepath):
            raise ValueError(f"Filepath {filepath} does not exist.")
        elif filepath.endswith(".json"):
            with open(filepath) as f:
                _update_keys_memory(json.load(f))
        else:
            raise ValueError(
                f"File type of {filepath} is not supported (only .json)")

    # load api keys from keyword arguments
    elif set(kwargs.keys()).issubset(set(supported_apis)):
        _update_keys_memory(kwargs)
    else:
        raise ValueError(f"Only {supported_apis} apis are supported.")

    # save to cache and sync
    _save_keys_to_cache()
    _sync_keys_with_apis()


def load_keys_from_cache() -> int:
    """Load API keys from the disk cache."""
    load_dotenv(env_path)  # from cache to os environment
    num_keys = _load_keys_from_env()  #  load keys from os env
    logging.debug(f"Loaded {num_keys} api keys from cache"
                  if num_keys else "api keys env cache does not exist")


# --- private related to key management ---

def _update_keys_memory(keys: dict) -> None:
    """Update the API keys memory."""
    _verify_keys(keys)  #  check if keys are supported
    api_keys.update(keys)  #  then save to live memory


def _verify_keys(keys_from_env: dict) -> bool:
    """Verify that the API keys are supported."""
    for (api, key) in keys_from_env.items():
        if api not in supported_apis:
            raise ValueError(f"API {api} not supported.")


def _sync_keys_with_apis() -> None:
    """Update the API keys from the environment."""
    import openai
    logging.debug(f"Setting OpenAI API key to {api_keys.get('openai')}")
    openai.api_key = api_keys.get("openai")


# --- private related to key cache ---

def _save_keys_to_cache(filepath: str = env_path) -> None:
    """Save API keys to the disk cache."""
    _verify_keys(api_keys)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for api, key in api_keys.items():
            f.write(f"{api.upper()}=\"{key}\"\n")


def _load_keys_from_env() -> int:
    """Load API keys from os environment."""
    keys_from_cache = {
        key: os.environ.get(key.upper())
        for key in supported_apis
        if os.environ.get(key.upper())
    }
    _update_keys_memory(keys_from_cache)
    _sync_keys_with_apis()
    return len(keys_from_cache)
