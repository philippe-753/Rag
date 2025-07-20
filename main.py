import os
from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from dotenv import load_dotenv

load_dotenv()

SYSTEM_MESSAGE = "You are a helpful assistant and expert on AI safety. You will be given a question and your task is to answer it using the informaiton that you know about AI safety perticular."

def initial_message() -> None:
    print("I am a helpful assistant and expert on AI safety.")
    print("Please enter a question (type 'exit' to quit):")
    
def set_up_model(model_name:str) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    return ChatOpenAI(openai_api_key=api_key, model=model_name)

def ask_model(chat:ChatOpenAI, prompt:str, messages:list[str]):
    messages.append(HumanMessage(content=prompt))
    return chat.invoke(messages)

def update_chat(AI_response:str, messages:list[str]) -> list[str]:
    messages.append(AIMessage(content=AI_response))
    return messages

def main():
    model_name = "gpt-3.5-turbo-0125"
    chat = set_up_model(model_name)
    initial_message() 
    messages = [SystemMessage(content=SYSTEM_MESSAGE)]

    while True:
        prompt = input(">> ")
        if prompt.lower() == "exit" or prompt.lower() == "exit()":
            break
        
        res = ask_model(chat, prompt, messages)
        print(f"{res.content}\n")

        messages = update_chat(AI_response=res.content, messages=messages)

if __name__ == "__main__":
    main()