from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from sustainabot import SustainabotAgent, LogWriter

app = FastAPI()

# CORS für lokale Entwicklung
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globale Instanz des AnimalAgent
agent = SustainabotAgent()
log_writer = LogWriter()

class ChatMessage(BaseModel):
    message: str
    chat_history: List[str] = []

class ChatResponse(BaseModel):
    response: str
    state: str
    log_message: Dict[str, Any]

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    try:
        response, log_message = agent.get_response(
            chat_message.message, 
            chat_message.chat_history
        )
        log_writer.write(log_message)
        return ChatResponse(
            response=response,
            state=response,
            log_message=log_message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat") 
async def chat(request: Request): 
    data = await request.json() 
    user_message = data["message"]
    chat_history = data.get("chat_history", [])
    response_text, log_message = agent.get_response(user_message, chat_history)
    log_writer.write(log_message) #for saving information about the conversation in conversation.jsonp
    return {
        "response": response_text,
        "state": agent.wrap_up_triggered
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)