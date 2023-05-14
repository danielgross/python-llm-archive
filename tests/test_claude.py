import unittest
import llm
import os
import vcr
import asynctest
import asyncio

llm.set_api_key(anthropic=open(
    os.path.expanduser("~/.anthropic")).read().strip())


class TestClaudeCompletions(unittest.TestCase):

    @vcr.use_cassette("tests/fixtures/claude/test_completion.yaml", filter_headers=['authorization', 'x-api-key'])
    def test_completion(self):
        prompt = "Hello, my name is"
        completion = llm.complete(
            prompt, engine="anthropic:claude-v1", max_tokens_to_sample=1).strip()
        self.assertTrue(completion.startswith("Claude"))


    @vcr.use_cassette("tests/fixtures/claude/test_chat.yaml", filter_headers=['authorization', 'x-api-key'])
    def test_chat(self):
        messages = ["what is your favorite cat breed?", "I like tabbies.", "And what about dogs? Reply with punctuation."]
        response = llm.chat(messages, engine="anthropic:claude-v1")
        self.assertTrue(response[0] != " ", "Response should not start with a space.")
        # Make sure the response is sorta directionally correct, e.g. has more than 3 words and some punctuation.
        self.assertTrue(len(response.split(' ')) > 3)
        self.assertTrue(
            '.' in response or '!' in response or '?' in response or ',' in response, response)
        

class TestClaudeStreaming(asynctest.TestCase):

    @vcr.use_cassette("tests/fixtures/claude/test_streaming_chat.yaml", filter_headers=['authorization', 'x-api-key'])
    async def test_streaming_chat(self):    
        messages = ["what is your favorite cat breed?", "I like tabbies.", "And what about dogs? Reply with punctuation."]
        responses = []
        async for response in llm.stream_chat(messages, engine="anthropic:claude-v1"):
            responses.append(response)
        final_response = responses[-1]
        self.assertTrue(response[0] != " ", "Response should not start with a space.")
        # Make sure we're not repeating ourselves within this final response.
        self.assertTrue(final_response[:5] not in final_response[5:])
        self.assertTrue(len(final_response.split(' ')) > 3)
        self.assertTrue(
            '.' in response or '!' in response or '?' in response, response)