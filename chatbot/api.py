from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from sustainabot import SustainabilityConsultant, load_faiss_index_and_docs

app = FastAPI()


# CORS für lokale Entwicklung
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS index and docs ONCE for the API process
index, docs = load_faiss_index_and_docs()
agent = SustainabilityConsultant()

class ChatMessage(BaseModel):
    message: str
    chat_history: List[str] = []

class ChatResponse(BaseModel):
    response: str
    log_message: Dict[str, Any]

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    try:
        # Check for goodbye message
        if hasattr(agent, "is_goodbye_message") and agent.is_goodbye_message(chat_message.message):
            goodbye_text = "🌱 Thank you for using the Sustainable Packaging Consultant! Have a green day! 🌎"
            return ChatResponse(
                response=goodbye_text,
                log_message={"info": "User ended chat with goodbye message."}
            )
        # Pass index and docs to the consultant for every request
        response, log_message = agent.get_response(
            chat_message.message, 
            chat_message.chat_history,
            index,
            docs
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