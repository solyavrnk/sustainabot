from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sustainabot import SustainabilityConsultant, load_faiss_index_and_docs
from log_writer import LogWriter

app = FastAPI()


# CORS für lokale Entwicklung
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary für Session-spezifische Agenten
session_agents: Dict[str, SustainabilityConsultant] = {}

# Load FAISS index and docs ONCE for the API process
index, docs = load_faiss_index_and_docs()
agent = SustainabilityConsultant()
# Instantiate log writer and attach to agent
agent.log_writer = LogWriter()

class ChatMessage(BaseModel):
    session_id: str
    message: str
    chat_history: List[str] = []
    generate_roadmap: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    is_loading: bool = False  # default False
    log_message: Dict[str, Any]
    roadmap: Optional[List[str]] = None  

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    
    # Prüfen ob eine Session-ID existiert, sonst neue erstellen
    if chat_message.session_id not in session_agents:
        session_agents[chat_message.session_id] = SustainabilityConsultant()
        
        agent = session_agents[chat_message.session_id]
        response, log_message = agent.get_response(
            chat_message.message, 
            chat_message.chat_history
        )
          
    # At this point consent_given is True → proceed as usual
    try:
        user_message = chat_message.message.strip().lower()

        # Begrüßung
        if not user_message:
            if hasattr(agent, "generate_greeting"):
                greeting = agent.generate_greeting()
            else:
                greeting = "🌱 Hello! I'm your Sustainable Packaging Consultant. How can I help you today?"
            log_message = {"info": "Bot greeting"}
            # Log in Datei schreiben
            agent.log_writer.write(log_message)
            return ChatResponse(
                response=greeting,
                log_message=log_message
            )

        # Goodbye
        if hasattr(agent, "is_goodbye_message") and agent.is_goodbye_message(user_message):
            goodbye_text = "🌱 Thank you for using the Sustainable Packaging Consultant! Have a green day! 🌎"
            log_message = {"info": "User ended chat"}
            # Log in Datei schreiben
            agent.log_writer.write(log_message)
            return ChatResponse(
                response=goodbye_text,
                log_message=log_message
            )

        response, is_loading, log_message, roadmap = agent.get_response(
            user_message,
            chat_message.chat_history,
            index,
            docs,
            generate_roadmap=chat_message.generate_roadmap
        )
        # zieh Dir hier den aktuellen Slot‐State aus dem Agenten
        log_message["slots"] = {
            k: (v if v is not None else "")
            for k, v in agent.slots.slots.items()
        }

        if hasattr(agent, "log_writer"):
            agent.log_writer.write(log_message)

         # If roadmap is still generating, override response with loading message
        if is_loading:
            response = "🛠️ Roadmap is being created... This might take a moment ⏳"

        return ChatResponse(
            response=response,
            is_loading=is_loading,
            log_message=log_message,
            roadmap = roadmap
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)