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
    consent_given: Optional[bool] = None  # <-- make optional, default None

class ChatResponse(BaseModel):
    response: str
    is_loading: bool = False  # default False
    log_message: Dict[str, Any]

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    if chat_message.consent_given is False:
        # Explicit denial of consent
        return ChatResponse(
            response="❌ Consent not given. You cannot use this service. Stay sustainable! 🌿",
            log_message={"error": "Consent explicitly denied by user"}
        )

    if chat_message.consent_given is None:
        # Consent not provided yet — send explanation + prompt
        consent_request_text = (
            "\n\n"
            "Before we get started: \n\n"
            "This chatbot will process your input to provide tailored sustainability advice.\n"
            "Your input may be logged for improving the service, but no personal data is stored‼️\n\n"
            "🔐 Do you agree to continue? Please reply with 'yes' or 'no'."
        )
        return ChatResponse(
            response=consent_request_text,
            log_message={"info": "Consent requested"}
        )
    
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

        if user_message in ["no", "n"]:
            # User explicitly said no after consent request
            return ChatResponse(
                response="❌ Consent not given. Stay sustainable! 🌿",
                log_message={"error": "Consent explicitly denied by user after prompt"}
            )

        if user_message in ["yes", "y"]:
            greeting_response, is_loading, log_message = agent.get_response(
                user_question="hello",  # triggers greeting logic in agent
                chat_history=chat_message.chat_history,
                index=index,
                docs=docs
            )
            return ChatResponse(
                response=f"🔐 Consent approved.\n\n{greeting_response}",
                is_loading=is_loading,
                log_message={**log_message, "info": "Consent given by user after prompt"}
            )

        # Normal chat flow
        if hasattr(agent, "is_goodbye_message") and agent.is_goodbye_message(user_message):
            goodbye_text = "🌱 Thank you for using the Sustainable Packaging Consultant! Have a green day! 🌎"
            return ChatResponse(
                response=goodbye_text,
                log_message={"info": "User ended chat with goodbye message."}
            )

        response, is_loading, log_message = agent.get_response(
            user_message,
            chat_message.chat_history,
            index,
            docs
        )

        print("DEBUG: Sending response:", {
            "response": response,
            "is_loading": is_loading,
            "log_message": log_message
        })

        return ChatResponse(
            response=response,
            is_loading=is_loading,
            log_message=log_message
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)