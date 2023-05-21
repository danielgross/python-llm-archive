import unittest
import asynctest
import os
import vcr
import llm
import llm.api.openaiapi
import logging

# Only get API keys if they exist
_key = open(os.path.expanduser("~/.openai")).read().strip() if os.path.exists(
    os.path.expanduser("~/.openai")) else 'test'
llm.set_api_key(openai=_key)


class TestOpenAICompletions(unittest.TestCase):

    @vcr.use_cassette("tests/fixtures/openai/test_completion.yaml", filter_headers=['authorization'])
    def test_completion(self):
        prompt = "2+2 is how much? Good question, I think 2+2="
        completion = llm.complete(
            prompt, engine="openai:text-davinci-002", max_tokens=1)
        # Assert result is 4 or 5 because sometimes OpenAI returns 2+2=5
        self.assertTrue(completion in ["4", "5"], completion)

    @vcr.use_cassette("tests/fixtures/openai/test_simple_chat.yaml", filter_headers=['authorization', 'x-api-key'])
    def test_simple_chat(self):
        self.assertEqual(llm.chat(
            "What is 2+2? Reply with just one number and no punctuaction.", engine="openai:gpt-3.5-turbo"), "4")
        self.assertEqual(llm.chat(
            ["What is 2+2? Reply with just one number and no punctuaction."], engine="openai:gpt-3.5-turbo"), "4")

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
        

    @vcr.use_cassette("tests/fixtures/openai/test_token_overflow.yaml", filter_headers=['authorization'])
    def test_token_overflow(self):
        """Test that we can handle a token overflow."""
        prompt = "Answer basic math question. 2+2=X Then multiply X by 100. Final answer is?"
        normal_completion = llm.complete(prompt=prompt, engine="openai:gpt-3.5-turbo")
        chat_completion = llm.chat(prompt, engine="openai:gpt-3.5-turbo")
        self.assertTrue("400" in normal_completion, normal_completion)
        self.assertTrue("400" in chat_completion, chat_completion)
        llm.api.openaiapi.MODEL_CONTEXT_SIZE_LIMITS["gpt-3.5-turbo"] = 10
        normal_completion = llm.complete(prompt=prompt, engine="openai:gpt-3.5-turbo", prompt_overflow="trim")
        chat_completion = llm.chat(prompt, engine="openai:gpt-3.5-turbo", prompt_overflow="trim")
        self.assertTrue("4" in normal_completion and "400" not in normal_completion, normal_completion)
        self.assertTrue("4" in chat_completion and "400" not in chat_completion, chat_completion)


class TestOpenAIStreaming(asynctest.TestCase):

    @vcr.use_cassette("tests/fixtures/openai/test_streaming_chat_method_full.yaml", filter_headers=['authorization', 'x-api-key'])
    async def test_streaming_chat_method_full(self):
        messages = ["what is your favorite cat breed?", "I like tabbies.",
                    "And what about dogs?"]
        responses = []
        async for response in llm.stream_chat(messages, engine="openai:gpt-3.5-turbo", stream_method="full"):
            responses.append(response)
        final_response = responses[-1]
        self.assertEqual("As an AI language model, I do not have preferences or emotions. However, I can provide you some information about different dog breeds if you would like to know.",
                         final_response, final_response)

    @vcr.use_cassette("tests/fixtures/openai/test_streaming_chat_method_delta.yaml", filter_headers=['authorization', 'x-api-key'])
    async def test_streaming_chat_method_delta(self):
        messages = ["what is your favorite cat breed?", "I like tabbies.",
                    "And what about dogs?"]
        responses = []
        async for response in llm.stream_chat(messages, engine="openai:gpt-3.5-turbo", stream_method="delta"):
            responses.append(response)
        self.assertEqual(responses[0], 'As')
        self.assertEqual(responses[1], ' an')
