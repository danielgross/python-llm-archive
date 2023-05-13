import requests
import anthropic
from llm.util import api_keys


BASE_URL = "https://api.anthropic.com"


def complete(prompt, engine="claude-v1", max_tokens_to_sample=10, **kwargs):
    headers = {
        "x-api-key": api_keys.get("anthropic"),
        "Content-Type": "application/json",
    }

    data = {
        "prompt": prompt,
        "model": engine,
        "max_tokens_to_sample": max_tokens_to_sample,
        **kwargs,
    }

    response = requests.post(
        f"{BASE_URL}/v1/complete",
        headers=headers,
        json=data,
    )

    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")

    result = response.json()
    return result["completion"]


def chat(messages, engine="claude-v1", system=None, max_tokens_to_sample=30, **kwargs):
    if system is not None:
        raise NotImplementedError(
            "System messages are not yet implemented for Claude.")

    c = anthropic.Client(api_keys["anthropic"])
    prompt = []
    for message in messages:
        if message["role"] == "user":
            prompt.append(f"{anthropic.HUMAN_PROMPT} {message['content']}")
        elif message["role"] == "assistant":
            prompt.append(f"{anthropic.AI_PROMPT} {message['content']}")
        else:
            raise ValueError(f"Unknown role {message['role']}")
    if messages[-1]["role"] != "user":
        raise ValueError("Last message must be from the user.")
    # Add a final AI prompt
    prompt.append(anthropic.AI_PROMPT)
    print(prompt)
    prompt = "".join(prompt)
    resp = c.completion(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model=engine,
        max_tokens_to_sample=max_tokens_to_sample,
        **kwargs
    )
    return resp["completion"]
