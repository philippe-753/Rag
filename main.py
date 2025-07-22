import os
from dotenv import load_dotenv
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import TokenTextSplitter


load_dotenv()

SYSTEM_MESSAGE: str = (
    "You are a helpful assistant and expert on AI safety. "
    "Use ONLY the provided context to answer the question. "
    "If the context is not relevant or missing, respond with 'I don't know'."
)

# Load FAISS vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


def retrieve_context(query: str, messages: List[BaseMessage], k: int = 10, max_tokens: int = 1500) -> str:
    # Build a synthetic query using recent history
    recent_history = "\n".join([
        f"User: {msg.content}" for msg in messages[-4:] if isinstance(msg, HumanMessage)
    ])
    full_query = recent_history + f"\nCurrent question: {query}"

    docs = vector_store.similarity_search(full_query, k=k)
    context = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    ])
    
    splitter = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
    chunks = splitter.split_text(context)
    return chunks[0] if chunks else ""


def initial_message() -> None:
    print("I am a helpful assistant and expert on AI safety.")
    print("Please enter a question (type 'exit' to quit):")


def set_up_model(model_name: str) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    return ChatOpenAI(openai_api_key=api_key, model=model_name)


def ask_model(chat: ChatOpenAI, query: str, context: str, messages: List[BaseMessage]) -> AIMessage:
    messages.append(HumanMessage(content=(
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )))
    return chat.invoke(messages)


def update_chat(AI_response: str, messages: List[BaseMessage]) -> List[BaseMessage]:
    messages.append(AIMessage(content=AI_response))
    if len(messages) > 10:
        messages = trim_messages(messages, 10)
    return messages


def trim_messages(messages: List[BaseMessage], num_messages: int) -> List[BaseMessage]:
    return [messages[0]] + messages[-(num_messages - 1):]


def main() -> None:
    model_name: str = "gpt-3.5-turbo-0125"
    chat: ChatOpenAI = set_up_model(model_name)
    initial_message()
    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_MESSAGE)]

    while True:
        query: str = input(">> ")
        if query.lower() in {"exit", "exit()"}:
            break

        context: str = retrieve_context(query, messages)
        res: AIMessage = ask_model(chat, query, context, messages)
        print(f"Answer: {res.content}\n")

        messages = update_chat(AI_response=res.content, messages=messages)


if __name__ == "__main__":
    main()