import requests
from llm.util import api_keys

BASE_URL = "https://api.anthropic.com"


def complete(prompt, model="claude-v1", **kwargs):
    headers = {
        "x-api-key": api_keys.get("anthropic"),
        "Content-Type": "application/json",
    }

    data = {
        "prompt": prompt,
        "model": model,
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