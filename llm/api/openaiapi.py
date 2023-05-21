"""Implement the OpenAI API."""

from llm.utils.parsing import structure_chat
import openai
import tiktoken
import logging

logger = logging.getLogger(__name__)

MODEL_CONTEXT_SIZE_LIMITS = {
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "text-davinci-003": 4097,
    "text-davinci-002": 4097,
    "code-davinci-002": 8001,
}


def _trim_overflow(text, model):
    """Trim text to max_tokens."""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) > MODEL_CONTEXT_SIZE_LIMITS[model]:
        tokens = tokens[:MODEL_CONTEXT_SIZE_LIMITS[model]]
        return enc.decode(tokens)
    return text


def complete(prompt, engine, prompt_overflow=None, **kwargs):
    """Complete text using the OpenAI API."""
    if prompt_overflow == "trim":
        logging.debug(f"Trimming overflow for {engine}")
        prompt = _trim_overflow(prompt, engine)

    if engine.startswith("gpt-3.5") or engine.startswith("gpt-4"):
        return chat(structure_chat([prompt]), engine=engine, **kwargs)
    logger.debug(f"Completing with {engine} using prompt: {prompt}")
    return openai.Completion.create(
        engine=engine,
        prompt=prompt,
        **kwargs
    ).choices[0].text


def chat(messages, engine, system=None, prompt_overflow=None, **kwargs):
    """Chat with the OpenAI API."""
    if prompt_overflow == "trim":
        logging.debug(f"Trimming overflow for {engine}")
        # TODO For now, just trim the last message.
        messages[-1]['content'] = _trim_overflow(
            messages[-1]['content'], engine)

    if engine.startswith("text-"):
        raise ValueError(
            f"Cannot issue a ChatCompletion with engine {engine}.")
    if system is not None:
        messages = [{"role": "system", "content": system}] + messages
    logger.debug(f"Chatting with {engine} using messages: {messages}")
    response = openai.ChatCompletion.create(
        model=engine,
        messages=messages,
        **kwargs
    )
    return response.choices[0].message.content


async def stream_chat(messages, engine, system=None, prompt_overflow=None, **kwargs):
    """Chat with the OpenAI API."""
    if prompt_overflow == "trim":
        logging.debug(f"Trimming overflow for {engine}")
        # TODO For now, just trim the last message.
        messages[-1]['content'] = _trim_overflow(
            messages[-1]['content'], engine)

    if system is not None:
        messages = [{"role": "system", "content": system}] + messages
    logger.debug(f"Chatting with {engine} using messages: {messages}")

    result = openai.ChatCompletion.create(
        model=engine,
        messages=messages,
        stream=True,
        **kwargs
    )
    for chunk in result:
        if chunk.choices[0].delta.get('content') != None:
            yield chunk.choices[0].delta.content
