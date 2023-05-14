import unittest
import os

import vcr
import llm


llm.set_api_key(openai=open(os.path.expanduser("~/.openai")).read().strip())


class TestOpenAICompletions(unittest.TestCase):

    @vcr.use_cassette("tests/fixtures/openai/test_completion.yaml", filter_headers=['authorization'])
    def test_completion(self):
        prompt = "Hello, my name is"
        completion = llm.complete(
            prompt, engine="openai:text-davinci-002", max_tokens=1)
        # Assert we only got one extra word that is capitalized
        self.assertEqual(len(completion.split(' ')), 1)
        self.assertTrue(completion[0].isupper(), completion)

    @vcr.use_cassette("tests/fixtures/openai/test_completion_chat_model.yaml", filter_headers=['authorization'])
    def test_completion_chat_model(self):
        """Test that we can use a chat model for completion."""
        prompt = "Hello, my name is"
        completion = llm.complete(
            prompt, engine="openai:gpt-3.5-turbo", max_tokens=1)
        # Assert we only got one extra word that is capitalized
        self.assertEqual(len(completion.split(' ')), 1)
        self.assertTrue(completion[0].isupper(), completion)

    @vcr.use_cassette("tests/fixtures/openai/test_chat.yaml", filter_headers=['authorization'])
    def test_chat(self):
        messages = ["hello", "how are you?", "I'm good, thanks."]
        response = llm.chat(messages, engine="openai:gpt-3.5-turbo")
        # Make sure the response is sorta directionally correct, e.g. has more than 3 words and some punctuation.
        self.assertTrue(len(response.split(' ')) > 3)
        self.assertTrue(
            '.' in response or '!' in response or '?' in response, response)
