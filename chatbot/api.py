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

from typing import Optional

class ChatMessage(BaseModel):
    message: str
    chat_history: List[str] = []
    generate_roadmap: Optional[bool] = False  # default False

class ChatResponse(BaseModel):
    response: str
    is_loading: bool = False
    log_message: Dict[str, Any]
    roadmap: Optional[List[str]] = None  # ✅ Add this


@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    # At this point consent_given is True → proceed as usual
    try:
        user_message = chat_message.message.strip().lower()

        if not user_message:
            if hasattr(agent, "generate_greeting"):
                greeting = agent.generate_greeting()
            else:
                greeting = "🌱 Hello! I'm your Sustainable Packaging Consultant. How can I help you today?"
            return ChatResponse(
                response=greeting,
                log_message={"info": "Bot started the conversation due to empty user message."}
            )

        # Normal chat flow
        if hasattr(agent, "is_goodbye_message") and agent.is_goodbye_message(user_message):
            goodbye_text = "🌱 Thank you for using the Sustainable Packaging Consultant! Have a green day! 🌎"
            return ChatResponse(
                response=goodbye_text,
                log_message={"info": "User ended chat with goodbye message."}
            )

        response, is_loading, log_message, roadmap = agent.get_response(
            user_message,
            chat_message.chat_history,
            index,
            docs,
            generate_roadmap=chat_message.generate_roadmap
        )

        # If roadmap is still generating, override response with loading message
        if is_loading:
            response = "🛠️ Roadmap is being created... This might take a moment ⏳"

        print("DEBUG: Sending response:", {
            "response": response,
            "is_loading": is_loading,
            "log_message": log_message
        })

        return ChatResponse(
            response=response,
            is_loading=is_loading,
            log_message=log_message,
            roadmap=roadmap  # ✅ use actual roadmap variable
        )


    except Exception as e:
        import traceback
        traceback.print_exc()
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)