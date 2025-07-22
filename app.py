from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel
from main import set_up_model, retrieve_context, ask_model, update_chat, SYSTEM_MESSAGE
from langchain.schema import SystemMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or set to your specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html
@app.get("/", response_class=HTMLResponse)
def get_home():
    return Path("static/index.html").read_text()

# Load RAG model
chat = set_up_model("gpt-3.5-turbo-0125")
messages = [SystemMessage(content=SYSTEM_MESSAGE)]

class Query(BaseModel):
    query: str

@app.post("/chat")
def chat_with_rag(data: Query):
    global messages
    query = data.query
    context = retrieve_context(query, messages)
    res = ask_model(chat, query, context, messages)
    messages = update_chat(res.content, messages)
    return {"response": res.content}
