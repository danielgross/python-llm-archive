"""Implement the OpenAI API."""

import openai
import logging

from llm.utils.parsing import structure_chat


def complete(prompt, engine="text-davinci-003", **kwargs):
    """Complete text using the OpenAI API."""
    if engine.startswith("gpt-3.5") or engine.startswith("gpt-4"):
        return chat(structure_chat([prompt]), engine=engine, **kwargs)
    logging.debug(f"Completing with {engine} using prompt: {prompt}")
    return openai.Completion.create(
        engine=engine,
        prompt=prompt,
        **kwargs
    ).choices[0].text.strip()


def chat(messages, system=None, engine="gpt-3.5-turbo", **kwargs):
    """Chat with the OpenAI API."""
    if system is not None:
        messages = [{"role": "system", "content": system}] + messages
    logging.debug(f"Chatting with {engine} using messages: {messages}")
    response = openai.ChatCompletion.create(
        model=engine,
        messages=messages,
        **kwargs
    )
    return response.choices[0].message.content.strip()
