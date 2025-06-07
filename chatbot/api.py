from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from sustainabot import SustainabotAgent

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
        return ChatResponse(
            response=response,
            state=response,
            log_message=log_message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)