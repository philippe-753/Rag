# tests/test_main.py
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from main import update_chat, ask_model, retrieve_context, trim_messages
from unittest.mock import MagicMock

class FakeChat:
    def invoke(self, messages):
        return AIMessage(content="Mock reply")

def test_retrieve_context_mock(monkeypatch):
    class DummyDoc:
        def __init__(self, content, source):
            self.page_content = content
            self.metadata = {"source": source}

    dummy_docs = [DummyDoc("AI safety matters", "doc1")]
    dummy_splitter = MagicMock()
    dummy_splitter.split_text.return_value = ["[Source: doc1]\nAI safety matters"]
    monkeypatch.setattr("main.vector_store.similarity_search", lambda q, k: dummy_docs)
    monkeypatch.setattr("main.TokenTextSplitter", lambda chunk_size, chunk_overlap: dummy_splitter)

    messages = [HumanMessage(content="What is alignment?")]
    result = retrieve_context("What is AI?", messages)
    assert result == "[Source: doc1]\nAI safety matters"


def test_update_chat():
    messages = [SystemMessage(content="Hello")]
    result = update_chat("Hi!", messages)
    assert isinstance(result[-1], AIMessage)
    assert result[-1].content == "Hi!"

def test_ask_model():
    messages = [SystemMessage(content="System message")]
    prompt = "What is AI safety?"
    context = "AI safety is important"
    chat = FakeChat()

    result = ask_model(chat, prompt, context, messages)
    assert isinstance(result, AIMessage)
    assert result.content == "Mock reply"
    assert isinstance(messages[-1], HumanMessage)

def test_context_retrieval():
    query = "What is AI safety?"
    messages = [SystemMessage(content="System message")]
    result = retrieve_context(query, messages)
    assert isinstance(result, str)

def test_trim_messages():
    messages = [SystemMessage(content="System")]
    messages += [HumanMessage(content=f"Q{i}") for i in range(10)]
    trimmed = trim_messages(messages, 5)
    assert len(trimmed) == 5
    assert isinstance(trimmed[0], SystemMessage)

def test_update_chat_trims():
    # Test with mode that 10 chats.
    messages = [SystemMessage(content="System")]
    for i in range(10):
        messages.append(HumanMessage(content=f"Q{i}"))
    updated = update_chat("Test", messages, 10)
    assert len(updated) == 10
    assert isinstance(updated[-1], AIMessage)
