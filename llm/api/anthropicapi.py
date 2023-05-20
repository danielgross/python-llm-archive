import requests
import anthropic

from llm.utils.apikeys import api_keys


BASE_URL = "https://api.anthropic.com"


def client():
    return anthropic.Client(api_keys["anthropic"])


def format_chat_messages(messages):
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
    return "".join(prompt)


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

    c = client()
    prompt = format_chat_messages(messages)
    resp = c.completion(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model=engine,
        max_tokens_to_sample=max_tokens_to_sample,
        **kwargs
    )
    return resp["completion"]


async def stream_chat(messages, engine="claude-v1", system=None, max_tokens_to_sample=100, **kwargs):
    c = client()
    prompt = format_chat_messages(messages)
    response = await c.acompletion_stream(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        max_tokens_to_sample=max_tokens_to_sample,
        model=engine,
        stream=True,
    )
    async for data in response:
        # return a generator of responses
        yield data["completion"]
