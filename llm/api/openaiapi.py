"""Implement the OpenAI API."""

from llm.utils.parsing import structure_chat
import openai
import tiktoken
import logging

logger = logging.getLogger(__name__)


def _trim_overflow(text, model, max_tokens):
    """Trim text to max_tokens."""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        return enc.decode(tokens)
    return text


def complete(prompt, engine, max_prompt_tokens=None, **kwargs):
    """Complete text using the OpenAI API."""
    if max_prompt_tokens:
        logging.debug(f"Trimming overflow for {engine}")
        prompt = _trim_overflow(prompt, engine, max_prompt_tokens)

    if engine.startswith("gpt-3.5") or engine.startswith("gpt-4"):
        return chat(structure_chat([prompt]), engine=engine, **kwargs)
    logger.debug(f"Completing with {engine} using prompt: {prompt}")
    return openai.Completion.create(
        engine=engine,
        prompt=prompt,
        **kwargs
    ).choices[0].text


def chat(messages, engine, system=None, max_prompt_tokens=None, **kwargs):
    """Chat with the OpenAI API."""
    if max_prompt_tokens:
        logging.debug(f"Trimming overflow for {engine}")
        # TODO For now, just trim the last message.
        messages[-1]['content'] = _trim_overflow(messages[-1]['content'], engine, max_prompt_tokens)

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


async def stream_chat(messages, engine, system=None, max_prompt_tokens=None, **kwargs):
    """Chat with the OpenAI API."""
    if max_prompt_tokens:
        logging.debug(f"Trimming overflow for {engine}")
        # TODO For now, just trim the last message.
        messages[-1]['content'] = _trim_overflow(messages[-1]['content'], engine, max_prompt_tokens)

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
