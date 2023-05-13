import unittest
import llm
import os

llm.set_api_key(anthropic=open(
    os.path.expanduser("~/.anthropic")).read().strip())


class TestClaudeCompletions(unittest.TestCase):

    def test_completion(self):
        prompt = "Hello, my name is"
        completion = llm.complete(
            prompt, engine="anthropic:claude-v1", max_tokens_to_sample=1).strip()
        self.assertTrue(completion.startswith("Claude"))
