# python-llm: A LLM API for Humans 
```python
import llm
llm.complete("hello, I am a") # Frog. By default uses TurboGPT.
llm.complete("hello, I am ", engine="anthropic:claude-instant-v1") # Goat.
llm.chat(["hi", "hi how are you", "tell me a joke"])
llm.embed(open("harrypotter.txt").read())
```