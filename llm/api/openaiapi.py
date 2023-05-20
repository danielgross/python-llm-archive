"""Implement the OpenAI API."""

from llm.utils.parsing import structure_chat
import openai
import logging

logger = logging.getLogger(__name__)


def complete(prompt, engine, **kwargs):
    """Complete text using the OpenAI API."""
    if engine.startswith("gpt-3.5") or engine.startswith("gpt-4"):
        return chat(structure_chat([prompt]), engine=engine, **kwargs)
    logger.debug(f"Completing with {engine} using prompt: {prompt}")
    return openai.Completion.create(
        engine=engine,
        prompt=prompt,
        **kwargs
    ).choices[0].text


def chat(messages, engine, system=None, **kwargs):
    """Chat with the OpenAI API."""
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


async def stream_chat(messages, engine, system=None, **kwargs):
    """Chat with the OpenAI API."""
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
