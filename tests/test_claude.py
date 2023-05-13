import unittest
import llm
import os
import requests

llm.set_api_key(anthropic=open(
    os.path.expanduser("~/.anthropic")).read().strip())


class TestClaudeCompletions(unittest.TestCase):

    def test_completion(self):
        prompt = "Hello, my name is"
        completion = llm.complete(
            prompt, engine="anthropic:claude-v1", max_tokens_to_sample=1).strip()
        self.assertTrue(completion.startswith("Claude"))

    def test_chat(self):
        messages = ["what is your favorite cat breed?", "I like tabbies.", "And what about dogs? Reply with punctuation."]
        response = llm.chat(messages, engine="anthropic:claude-v1")
        # Make sure the response is sorta directionally correct, e.g. has more than 3 words and some punctuation.
        self.assertTrue(len(response.split(' ')) > 3)
        self.assertTrue(
            '.' in response or '!' in response or '?' in response, response)