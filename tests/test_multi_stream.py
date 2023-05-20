import unittest
import llm
import os
import vcr
import asynctest
import collections


class TestAnthropicStreaming(asynctest.TestCase):

    @vcr.use_cassette("tests/fixtures/multi_stream/test_multistream.yaml", filter_headers=['authorization', 'x-api-key'])
    async def test_multistream(self):
        results = collections.defaultdict(list)
        async for engine, response in llm.multi_stream_chat(["Write a poem about the number pi."], engines=["anthropic:claude-instant-v1", "openai:gpt-3.5-turbo"], max_tokens=10):
            results[engine].append(response)
        final = {k: "".join(v[-1]) for k, v in results.items()}
        self.assertTrue(final["anthropic:claude-instant-v1"].startswith(
            "Here is a poem I wrote about the number pi"), final["anthropic:claude-instant-v1"])
        self.assertTrue(
            final["openai:gpt-3.5-turbo"].startswith("Pi, the circle’s constant"), final["openai:gpt-3.5-turbo"])
