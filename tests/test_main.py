# tests/test_main.py
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from main import update_chat, ask_model

class FakeChat:
    def invoke(self, messages):
        return AIMessage(content="Mock reply")

def test_update_chat():
    messages = [SystemMessage(content="Hello")]
    result = update_chat("Hi!", messages)
    assert isinstance(result[-1], AIMessage)
    assert result[-1].content == "Hi!"

def test_ask_model():
    messages = [SystemMessage(content="System message")]
    prompt = "What is AI safety?"
    chat = FakeChat()

    result = ask_model(chat, prompt, messages)
    assert isinstance(result, AIMessage)
    assert result.content == "Mock reply"
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == prompt
